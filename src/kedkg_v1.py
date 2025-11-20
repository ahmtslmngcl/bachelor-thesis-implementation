import os
import re
import time
import traceback
import collections
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Optional
import spacy
import openai

LINKING_FAIL_COUNT = 0

try: # lazy import
    from src.modeleditor import ModelEditor
    from src.queryexecutor import QueryExecutor
except ModuleNotFoundError:
    ModelEditor = None
    QueryExecutor = None


class KEDKG:
    
    ######################## INITIALIZATION AND ADAPTER CONSTRUCTION ########################
    def __init__(self, model_name, editor_name, llm_cfg, extra_prints=False):

        ### KB state
        self.qid2name = collections.defaultdict(str)
        self.triples = collections.defaultdict(set)
        self.qid_relations = collections.defaultdict(set)

        ### configs
        self.model_name = model_name
        self.editor_name = editor_name
        self.llm_cfg = llm_cfg
        self.client = None          # set in prepare()
        self.divide_prompt = None   # set in prepare()
        self.answer_prompt = None   # set in prepare()
        

    def prepare(self, cfg):
        global DEVICE, ENTITY, REL
        global tokenizer, model, bert_tokenizer, bert_model, entity_bert_tokenizer, entity_bert_model
        global nlp, extra_prints_flag
        global LLM_ANSWER_MODEL, LLM_DIVIDE_MODEL, LLM_BASE_URL, LLM_API_KEY

        DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        ENTITY = 0
        REL = 1

        llm_cfg = self.llm_cfg
        extra_prints_flag = bool(cfg.get("extra_prints", False))

        LLM_ANSWER_MODEL = llm_cfg.get("answer_model")
        LLM_DIVIDE_MODEL = llm_cfg.get("divide_model")
        LLM_BASE_URL     = llm_cfg.get("base_url", "http://127.0.0.1:11434/v1")
        LLM_API_KEY      = llm_cfg.get("api_key", "dummy")

        try:
            self.client = openai
            openai.api_base = LLM_BASE_URL
            openai.api_key = LLM_API_KEY
        except Exception as e:
            self.client = None
            print(f"Warning: Could not initialize OpenAI client: {e}")
            
        self.repo_root = os.getcwd()
        kedkg_root = os.path.join(self.repo_root, "external", "kedkg")
        model_dir  = os.path.join(kedkg_root, "model", "rebel-large")
        rel_ckpt   = os.path.join(kedkg_root, "train", "results", "best_model")
        ent_ckpt   = os.path.join(kedkg_root, "train", "results_entity_judge", "best_model_entity_judge")
        nlp_dir    = os.path.join(kedkg_root, "en_core_web_md")
        prompts     = os.path.join(self.repo_root, "src", "prompts")

        # REBEL
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)

        # BERT relation/entity classifiers
        bert_tokenizer        = DistilBertTokenizer.from_pretrained(rel_ckpt)
        bert_model            = DistilBertForSequenceClassification.from_pretrained(rel_ckpt, num_labels=2).to(DEVICE)
        entity_bert_tokenizer = DistilBertTokenizer.from_pretrained(ent_ckpt)
        entity_bert_model     = DistilBertForSequenceClassification.from_pretrained(ent_ckpt, num_labels=2).to(DEVICE)

        # Prompts
        prompt_cfg = cfg.get("prompt_cfg")
        divide_file = prompt_cfg.get("divide_prompt")
        answer_file = prompt_cfg.get("answer_prompt")
        with open(os.path.join(prompts, divide_file), "r") as f:
            self.divide_prompt = f.read()
        with open(os.path.join(prompts, answer_file), "r") as f:
            self.answer_prompt = f.read()
        
        # spaCy + entity linker
        nlp = spacy.load(nlp_dir)
        nlp.add_pipe("entityLinker", last=True)

    def make_ripple_adapters(self):
        """Construct the RippleEdits adapters (QueryExecutor + ModelEditor) for this KEDKG instance."""
        if QueryExecutor is None or ModelEditor is None:
            raise ImportError("Ripple adapters require src.queryexecutor and src.modeleditor to be importable.")
        query_executor = KEDKGQueryExecutor(self, self.model_name)
        model_editor   = KEDKGModelEditor(self, query_executor)
        return query_executor, model_editor
    ######################## INITIALIZATION AND ADAPTER CONSTRUCTION ########################


    ######################## CORE KEDKG FUNCTIONS (directly extracted from dynamic_exp.py) ########################
    def extract_relations_from_model_output(self, text):
        relations = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        text_replaced = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        for token in text_replaced.split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    relations.append({
                        'head': subject.strip(),
                        'type': relation.strip(),
                        'tail': object_.strip()
                    })
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            relations.append({
                'head': subject.strip(),
                'type': relation.strip(),
                'tail': object_.strip()
            })
        return relations

    class KB():
        def __init__(self):
            self.relations = []

        def are_relations_equal(self, r1, r2):
            return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

        def exists_relation(self, r1):
            return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

        def add_relation(self, r):
            if not self.exists_relation(r):
                self.relations.append(r)

        def print(self):
            print("Relations:")
            for r in self.relations:
                print(f"  {r}")
        
    def from_small_text_to_kb(self, text, verbose=False):
        kb = self.KB()

        # Tokenizer text
        model_inputs = tokenizer(text, max_length=512, padding=True, truncation=True,
                                return_tensors='pt').to(DEVICE)
        # if verbose:
        #     print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

        # Generate
        gen_kwargs = {
            "max_length": 216,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 3
        }
        generated_tokens = model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # create kb
        for sentence_pred in decoded_preds:
            relations = self.extract_relations_from_model_output(sentence_pred)
            for r in relations:
                kb.add_relation(r)

        return kb

    """    
    def link_entity(self, entity, nlp):
        pattern = r'Q\d+'
        name = entity
        try:
            linking = re.search(pattern, str(nlp(entity.capitalize())._.linkedEntities))
        except:
            linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
        if linking:
            linking = linking.group(0) # Q?
        else:
            linking = name
        return linking
    """

    def link_entity(self, entity, nlp):
        pattern = r'Q\d+'
        name = entity
        try:
            try:
                linking = re.search(pattern, str(nlp(entity.capitalize())._.linkedEntities))
            except:
                linking = re.search(pattern, str(nlp(entity)._.linkedEntities))
            if linking:
                linking = linking.group(0) # Q?
            else:
                linking = name
            return linking
        except Exception as e:
            global LINKING_FAIL_COUNT
            LINKING_FAIL_COUNT += 1
            print(f"[Error] link_entity fail for '{entity}': {e}")
            return name

    def run_llm_answer(self, query):
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information. You must provide a short and accurate answer in one sentence. Do not repeat the question. The format is 'Answer:'."}]
        message_prompt = {"role":"user","content":query}
        messages.append(message_prompt)
        f = 0
        while(f == 0):
            
            try:
                response = self.client.ChatCompletion.create(
                        model=LLM_ANSWER_MODEL,
                        messages = messages,
                        frequency_penalty=0,
                        # temperature=0
                        presence_penalty=0
                )
                result = response["choices"][0]['message']['content']
                f = 1
            except:
                print("openai error, retry")
                traceback.print_exc()
                time.sleep(10)
        return result

    def predict(self, question, relation, FLAG):
        if FLAG == ENTITY:
            # inputs = entity_bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            inputs = entity_bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            outputs = entity_bert_model(**inputs)
        elif FLAG == REL:
            # inputs = bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            inputs = bert_tokenizer(question, relation, return_tensors='pt', truncation=True, padding='max_length').to(DEVICE)
            outputs = bert_model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class = logits.argmax().item()
        return predicted_class, logits, probabilities

    def get_scores(self, question, objects, FLAG):
        scores = []
        for object in objects:
            predicted_class, logits, probabilities = self.predict(question, object, FLAG)
            score = probabilities[0, 1].item()
            if score < 0.5:
                continue
            scores.append((score, object))
        return scores

    def retrieve_from_graph(self, q):
        kb = self.from_small_text_to_kb(q, verbose=True)
        if extra_prints_flag and kb.relations:
            preview = "; ".join([f"{r['head']} —[{r['type']}]→ {r['tail']}" for r in kb.relations[:2]])
            tail = " …" if len(kb.relations) > 2 else ""
            print(f"[Model] ents    | {preview}{tail}")
        kb = kb.relations
        # choose entity

        entity_set = set()
        for k in kb:
            # capitalize
            entity_set.add(k['head'])
            entity_set.add(k['type'])
            entity_set.add(k['tail'])
        if len(entity_set) == 0:
            return None
        scores = self.get_scores(q, entity_set, ENTITY)
        if len(scores) == 0:
            return None
        scores.sort(key=lambda x: x[0], reverse=True)
        if extra_prints_flag:
            print(f"[Model] score   | entity top={scores[0][1]} p={scores[0][0]:.3f}")
        head = scores[0][1]
        # choose rels

        head_linking = self.link_entity(head, nlp)
        tmp_rels = self.qid_relations[head_linking]
        if len(tmp_rels) == 0:
            return None
        scores = self.get_scores(q, tmp_rels, REL)
        if len(scores) == 0:
            return None
        scores.sort(key=lambda x: x[0], reverse=True)
        if extra_prints_flag:
            print(f"[Model] score   | rel    top={scores[0][1]} p={scores[0][0]:.3f}")
        rel = scores[0][1]
        # find rels
        if rel in tmp_rels:
            # get all posible tail entitys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
            tail_linkings = list(self.triples[(head_linking, rel)])
            # return tail_linkings[0][1] # return sentence
            return self.qid2name[tail_linkings[0][0]] # return entity
        return None

    def verify_gold_path(self, entities, hops, path):
        if len(entities) != hops:
            return False
        for i in range(hops):
            if entities[i].lower() == path[i]["answer"].lower() or path[i]["answer"].lower() in entities[i].lower() or any(entities[i].lower() in p.lower() for p in path[i]["answer_alias"]):
                continue
            else:
                return False
        return True

    def run_llm_divide(self, query):
        messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
        message_prompt = {"role":"user","content":query}
        messages.append(message_prompt)
        f = 0
        while(f == 0):
            try:
                response = self.client.ChatCompletion.create(
                        model=LLM_DIVIDE_MODEL,
                        messages = messages,
                        # temperature=temperature,
                        # max_tokens=max_tokens,
                        frequency_penalty=0,
                        presence_penalty=0
                )
                result = response["choices"][0]["message"]["content"]
                f = 1
            except:
                print("openai error, retry")
                traceback.print_exc()
                time.sleep(10)
        return result

    def construct_graph(self, batch, qid2name, triples, qid_relations, nlp):
        for i in tqdm(range(len(batch))):
            d = batch[i]
            edits = d["requested_rewrite"]
            for edit in edits:
                # capitalize
                sentence = edit["prompt"].format(edit["subject"]) + ' ' + edit["target_new"]["str"]
                text = sentence
                kb = self.from_small_text_to_kb(text, verbose=True)
                # print(kb.relations)
                pattern = r'Q\d+'
                for triple in kb.relations:
                    # capitalize
                    head = triple['head']
                    tail = triple['tail']
                    rel = triple['type']
                    if head == None or head == "" or tail == None or tail == "" or rel == None or rel == "":
                        continue

                    # erase true answer
                    if edit["target_true"]["str"] in head or edit["target_true"]["str"] in tail:
                        continue
                    
                    try:
                        head_linking = re.search(pattern, str(nlp(head.capitalize())._.linkedEntities))
                    except:
                        head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))

                    try:
                        tail_linking = re.search(pattern, str(nlp(tail.capitalize())._.linkedEntities))
                    except:
                        tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
                    if head_linking:
                        head_linking = head_linking.group(0) # Q?
                    else:
                        head_linking = head
                    
                    # print(head_linking)
                    
                    if head_linking not in qid2name:
                        qid2name[head_linking] = head
                    
                    
                    
                    if tail_linking:
                        tail_linking = tail_linking.group(0)
                    else:
                        tail_linking = tail

                    if tail_linking not in qid2name:
                        qid2name[tail_linking] = tail
                    
                    triples[(head_linking, rel)].add((tail_linking, sentence))
                    qid_relations[head_linking].add(rel)       

    def modify_graph(self, d, qid2name, triples, qid_relations, nlp):
        edits = d["requested_rewrite"]
        for edit in edits:
            sentence = edit["prompt"].format(edit["subject"]) + ' ' + edit["target_new"]["str"]
            text = sentence
            kb = self.from_small_text_to_kb(text, verbose=True)
            pattern = r'Q\d+'

            clear_flag = collections.defaultdict(int)
            for triple in kb.relations:
                head = triple['head']
                tail = triple['tail']
                rel = triple['type']
                if head == None or head == "" or tail == None or tail == "" or rel == None or rel == "":
                    continue

                # erase true answer
                if edit["target_true"]["str"] in head or edit["target_true"]["str"] in tail:
                    continue
                
                try:
                    head_linking = re.search(pattern, str(nlp(head.capitalize())._.linkedEntities))
                except:
                    head_linking = re.search(pattern, str(nlp(head)._.linkedEntities))

                try:
                    tail_linking = re.search(pattern, str(nlp(tail.capitalize())._.linkedEntities))
                except:
                    tail_linking = re.search(pattern, str(nlp(tail)._.linkedEntities))
                if head_linking:
                    head_linking = head_linking.group(0) # Q?
                else:
                    head_linking = head
                
                # print(head_linking)
                
                if head_linking not in qid2name:
                    qid2name[head_linking] = head
                
                
                if tail_linking:
                    tail_linking = tail_linking.group(0)
                else:
                    tail_linking = tail

                if tail_linking not in qid2name:
                    qid2name[tail_linking] = tail
                
                # remove origin rel
                if clear_flag[(head_linking, rel)] == 0:
                    triples[(head_linking, rel)].clear()
                    # set flag
                    clear_flag[(head_linking, rel)] = 1
                    
                
                triples[(head_linking, rel)].add((tail_linking, sentence))
                qid_relations[head_linking].add(rel)
    ######################## CORE KEDKG FUNCTIONS (directly extracted from dynamic_exp.py) ########################

    ######################## ADDITIONAL HELPER FUNCTIONS (logic adapted/indirectly extracted from dynamic_exp.py) ########################
    def kedkg_restore_kb(self):
        """KB Reset to EMPTY (logic extracted from dynamic_exp.py)"""
        self.qid2name = collections.defaultdict(str)
        self.triples = collections.defaultdict(set)
        self.qid_relations = collections.defaultdict(set)
        if extra_prints_flag:
            print("[Model] restore | kb=0 (cleared)")

    def kedkg_answer_question(self, question, return_all=False):
        entity = ""
        entity_list = []
        # prompt = self.divide_prompt.replace("<<<<QUESTION>>>>", question)
        # output = self.run_llm_divide(prompt)
        # print("running v1")
        question = question.strip()
        if extra_prints_flag:
            print(f"\n[Model] divide  | in: {question}  ->  out: {question}")

        self.print_kb()
        retrieve = self.retrieve_from_graph(question)
        if retrieve is None:
            prompt = self.answer_prompt.replace("<<<<QUESTION>>>>", question)
            # not found, pass to LLM
            output = self.run_llm_answer(prompt)
            match = re.search(r'Answer: (.*)', output)
            if match:
                entity = match.group(1)
            else:
                entity = output
            if extra_prints_flag:
                print(f"[Model] answer  | source=llm    value={entity}")
        else:
            entity = retrieve
            if extra_prints_flag:
                print(f"[Model] answer  | source=graph  value={entity}")
        entity_list.append(entity)
        return entity_list if return_all else entity
 
    def ripple_modify_graph(self, sentence: str, skip_label: Optional[str] = None):
        """Thin wrapper around modfiy_graph() for RippleEdits compaitbility"""
        d = {
            "requested_rewrite": [{
                "prompt": "{}", 
                "subject": "",
                "target_new":  {"str": sentence},
                "target_true": {"str": skip_label or ""}
            }]
        }
        self.modify_graph(d, self.qid2name, self.triples, self.qid_relations, nlp)
    ######################## ADDITIONAL HELPER FUNCTIONS (logic adapted/indirectly extracted from dynamic_exp.py) ########################

    ######################## OPTIONAL PRINTERS ########################
    def print_kb(self, resolve_labels: bool = True, show_provenance: bool = True):
        if not extra_prints_flag:
            return
        total = sum(len(t) for t in self.triples.values())
        print(f"[Model] kb      | triples={total}")
        for (head, rel), tails in self.triples.items():
            for (tail, provenance) in tails:
                h = self.qid2name.get(head, head) if resolve_labels else head
                t = self.qid2name.get(tail, tail) if resolve_labels else tail
                if show_provenance:
                    print(f"[Model] kb      | {h} —[{rel}]→ {t}  # {provenance}")
                else:
                    print(f"[Model] kb      | {h} —[{rel}]→ {t}")

    def print_example_header(self, fact):
        if not extra_prints_flag:
            return
        subj = getattr(fact, "get_subject_label", lambda: None)() or getattr(fact, "_subject_id", None)
        rel  = getattr(fact, "_relation", None)
        rel_name = getattr(rel, "name", None) if rel is not None else str(rel)
        new_lbl = getattr(fact, "get_target_label", lambda: None)()
        phrased = getattr(fact, "get_fact_phrased", lambda: None)()
        # compute KB size
        kb_total = 0
        for (_h,_r), tails in self.triples.items():
            kb_total += len(tails)

        def _kv(label: str, value):
            if not extra_prints_flag:
                return
            if value is None or value == "":
                return
            print(f"  • {label:<9}: {value}")

        print("\n" + "="*80)
        print("EXAMPLE")
        _kv("subject", subj)
        _kv("relation", rel_name)
        _kv("new tail", new_lbl)
        _kv("phrased", phrased)
        print(f"  • KB size   : {kb_total} triples (pre-edit)")
        print("-"*80)
    ######################## OPTIONAL PRINTERS ########################


######################## RIPPLEEDITS INTEGRATION ADAPTERS (QueryExecutor + ModelEditor) ########################
if ModelEditor is not None and QueryExecutor is not None:

    class KEDKGQueryExecutor(QueryExecutor):
        """
        RippleEdits QueryExecutor implementation for KEDKG.
        Calls the KEDKG QA pipeline (kedkg_answer_question).
        """
        def __init__(self, engine: KEDKG, model_name: str):
            super().__init__(model=None, tokenizer=None, device=None, send_to_device=False)
            self.engine = engine
            self.model_name = model_name
            self.last_executed_query = None  # for edit tracing

        def get_model_name(self):
            return self.model_name

        def _generate_text(self, prompt: str, length: int) -> str:
            return self.engine.kedkg_answer_question(prompt, return_all=False)

        def execute_query(self, query, answer_length: int = 30) -> bool:
            """
            Mirrors the base class logic while remembering the last query so the
            editor can infer the previous (gold) answer as skip_label.
            """
            prompt = self._prompt_context + query.get_query_prompt()
            model_answer = self._generate_text(prompt, len(prompt) + answer_length)
            model_answer = model_answer.replace(self._prompt_context, '', 1)
            ok = self._verify_answer(model_answer, query.get_answers())
            self.last_executed_query = query # remember for the editor (so we can derive skip_label)
            return ok

  
    class KEDKGModelEditor(ModelEditor):
        """
        RippleEdits ModelEditor implementation for KEDKG.
        Applies edits via engine.ripple_modify_graph() and resets via engine.kedkg_restore_kb().
        """
        def __init__(self, engine: KEDKG, query_executor: KEDKGQueryExecutor):
            super().__init__(query_executor)
            self.engine = engine

        def edit_model(self, fact):
            if extra_prints_flag:
                self.engine.print_example_header(fact)

            sentence = fact.get_fact_phrased()
            # Based on previous_query, set skip_label flag so erasing the true answer can be mirrored correctly in edit_kb()
            skip_label = None
            previous_query = getattr(self._query_executor, "last_executed_query", None)
            try:
                same_sr = previous_query and \
                          getattr(previous_query, "_subject_id", None) == getattr(fact, "_subject_id", None) and \
                          getattr(previous_query, "_relation", None)   == getattr(fact, "_relation", None)
                if same_sr:
                    answers = previous_query.get_answers()  # list[list[str]]
                    if answers and isinstance(answers[0], (list, tuple)) and answers[0]:
                        candidate_old = answers[0][0]
                        new_label = fact.get_target_label() if hasattr(fact, "get_target_label") else None
                        if not new_label or candidate_old != new_label:
                            skip_label = candidate_old
            except Exception:
                pass

            # Apply edit
            self.engine.ripple_modify_graph(sentence, skip_label=skip_label)

        def restore_model(self):
            self.engine.kedkg_restore_kb()
######################## RIPPLEEDITS INTEGRATION ADAPTERS (QueryExecutor + ModelEditor) ########################
