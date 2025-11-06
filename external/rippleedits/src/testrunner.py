from enum import Enum, auto

from benchmark import RecentlyAddedExample, CounterFactualExample
from testcase import TestCase
import time

# flip to True to print additional information
EXTRA_PRINTS = True

class TestResult(Enum):
    NOT_EXECUTED = auto()
    PASSED = auto()
    FAILED = auto()


class ExampleResult(Enum):
    EXECUTED = auto()
    EDIT_FAILED = auto()
    NEW_FACT_KNOWN = auto()
    PREV_FACT_UNKNOWN = auto()


class TestRunner:

    def __init__(self, query_executor, model_editor):
        self._query_executor = query_executor
        self._model_editor = model_editor

    ### HELPERS FOR PRINTING
    def _kb_size(self):
        eng = getattr(self._query_executor, "engine", None)
        if not eng or not hasattr(eng, "triples"):
            return None
        total = 0
        try:
            for (_h, _r), tails in eng.triples.items():
                total += len(tails)
        except Exception:
            return None
        return total

    def _fmt_fact(self, fact):
        if fact is None:
            return "<??>", "<?>", None, None, None
        subj = getattr(fact, "get_subject_label", lambda: None)() or getattr(fact, "_subject_id", None) or "<?>"
        rel  = getattr(fact, "_relation", None)
        rel_name = getattr(rel, "name", None) or str(rel)
        new_lbl = getattr(fact, "get_target_label", lambda: None)()
        prev    = getattr(fact, "previous_fact", None)
        prev_lbl = getattr(prev, "get_target_label", lambda: None)() if prev else None
        phrased = getattr(fact, "get_fact_phrased", lambda: None)()
        return subj, rel_name, prev_lbl, new_lbl, phrased

    def _print_query_verbose(self, title, q, ok):
        if not EXTRA_PRINTS:
            return
        try:
            prompt = q.get_query_prompt()
        except Exception:
            prompt = str(q)
        gold = None
        try:
            gold = q.get_answers()
        except Exception:
            pass
        print(f"\nQUERY OVERVIEW:")
        print(f"  • prompt : {prompt}")
        if gold is not None:
            print(f"  • gold   : {gold}")
        print(f"  • result : {'PASS' if ok else 'FAIL'}")

    ###

    def run_testcases(
        self, 
        example, 
        test_cases, 
        skip_edit=False, 
        skip_restore=False, 
        skip_preconditions=False,
    ):
        example_result = ExampleResult.EXECUTED
        test_results = {TestResult.NOT_EXECUTED: [], TestResult.PASSED: [], TestResult.FAILED: []}

        ### For Printing
        if EXTRA_PRINTS:
            fact = getattr(example, "fact", None)
            subj, rel_name, prev_lbl, new_lbl, phrased = self._fmt_fact(fact)
            print("\n" + "="*80)
            print(f"EXAMPLE")
            print(f"  • subject   : {subj}")
            print(f"  • relation  : {rel_name}")
            if prev_lbl is not None:
                print(f"  • prev tail : {prev_lbl}")
            if new_lbl is not None:
                print(f"  • new tail  : {new_lbl}")
            if phrased:
                print(f"  • phrased   : {phrased}")
            kb_before = self._kb_size()
            if kb_before is not None:
                print(f"  • KB size   : {kb_before} triples (pre-edit)")
            print("-"*80)
        ###

        # Check testcase conditions
        if not skip_preconditions:
            if EXTRA_PRINTS:
                print(f"\n[PRECONDITIONS]")
            for test_case in test_cases:
                for condition_query in test_case.get_condition_queries():
                    # print('\nExecuting condition query')
                    ok = self._query_executor.execute_query(condition_query)
                    self._print_query_verbose("  Condition", condition_query, ok)
                    if not ok:
                        test_results[TestResult.NOT_EXECUTED].append(test_case)
                        if EXTRA_PRINTS:
                            print("  → precondition failed; test marked NOT_EXECUTED")
                        break


        # Check if fact is known/unknown according to example type
        if EXTRA_PRINTS:
            print(f"\n[PRE-EDIT FACT CHECK]")
        if isinstance(example, RecentlyAddedExample):
            # print('\nExecuting fact check query')
            q = example.fact.get_fact_query()
            ok = self._query_executor.execute_query(q)
            self._print_query_verbose("  Fact check (pre-edit)", q, ok)
            if ok:
                example_result = ExampleResult.NEW_FACT_KNOWN
                if EXTRA_PRINTS:
                    print("  → NEW_FACT_KNOWN (fact already known before edit)")
        elif isinstance(example, CounterFactualExample):
            #print('\nExecuting fact check query')
            q = example.previous_fact.get_fact_query()
            ok = self._query_executor.execute_query(q)
            self._print_query_verbose("  Prev fact check (pre-edit)", q, ok)
            if not ok:
                example_result = ExampleResult.PREV_FACT_UNKNOWN
                if EXTRA_PRINTS:
                    print("  → PREV_FACT_UNKNOWN (previous fact not known before edit)")

        if self._model_editor is None:
            return example_result, test_results

        # Modify model
        if not skip_edit:
            if EXTRA_PRINTS:
                print(f"\n[EDIT]\n")
            t0 = time.time()
            self._model_editor.edit_model(example.fact)
            if EXTRA_PRINTS:
                dt = time.time() - t0
                kb_after = self._kb_size()
                if kb_after is not None:
                    print(f"\n[Edit] Applied in {dt:.2f}s. KB size now: {kb_after} triples.")
        # Test edit
        if EXTRA_PRINTS:
            print(f"\n[POST-EDIT FACT CHECK]")
        # print('\nExecuting fact check query')
        q = example.fact.get_fact_query()
        ok = self._query_executor.execute_query(q)
        self._print_query_verbose("  Fact check (post-edit)", q, ok)
        if not ok:
            example_result = ExampleResult.EDIT_FAILED
            if EXTRA_PRINTS:
                print("  → EDIT_FAILED (target fact not satisfied after edit)")

        # Test modified model
        if EXTRA_PRINTS:
            print(f"\n[AXIS TESTS]")
        for test_case in test_cases:
            if test_case not in test_results[TestResult.NOT_EXECUTED]:
                test_case_results = []
                for test_query in test_case.get_test_queries():
                    # print('\nExecuting test query')
                    ok = self._query_executor.execute_query(test_query)
                    self._print_query_verbose("    Test", test_query, ok)
                    test_case_results.append(ok)
                if test_case.get_test_condition() == TestCase.OR_TEST_CONDITION and True in test_case_results:
                    test_results[TestResult.PASSED].append(test_case)
                    if EXTRA_PRINTS:
                        print("  → TestCase PASSED (OR)")
                elif test_case.get_test_condition() == TestCase.AND_TEST_CONDITION and False not in test_case_results:
                    test_results[TestResult.PASSED].append(test_case)
                    if EXTRA_PRINTS:
                        print("  → TestCase PASSED (AND)")
                else:
                    test_results[TestResult.FAILED].append(test_case)
                    if EXTRA_PRINTS:
                        print("  → TestCase FAILED")

        # Restore model
        if not skip_restore:
            if EXTRA_PRINTS:
                print(f"\n[RESTORE]\n")
            t0 = time.time()
            self._model_editor.restore_model()
            if EXTRA_PRINTS:
                dt = time.time() - t0
                kb_after = self._kb_size()
                if kb_after is not None:
                    print(f"\n[Restore] Done in {dt:.2f}s. KB size now: {kb_after} triples.")

        if EXTRA_PRINTS:
            n_not = len(test_results[TestResult.NOT_EXECUTED])
            n_ok  = len(test_results[TestResult.PASSED])
            n_bad = len(test_results[TestResult.FAILED])
            print("\nSUMMARY")
            print(f"  • example_result      : {example_result.name}")
            print(f"  • testcases_passed    : {n_ok}")
            print(f"  • testcases_failed    : {n_bad}")
            print(f"  • testcases_not_exec  : {n_not}")
            print("="*80)

        return example_result, test_results
