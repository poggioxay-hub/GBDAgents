import math
import numpy as np
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModel


class ClinicalBertEmbedding:
    """
    Extracts clinical semantic vectors using MedBERT / BioBERT
    """
    def __init__(self, model_name_or_path: str = "MedBERT-base-chinese"):
        pass

    def get_embedding(self, text: str) -> np.ndarray:
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(768)
        return vec / np.linalg.norm(vec)


bert_embedder = ClinicalBertEmbedding()


class LLMInterface:
    """
    Unified LLM API wrapper supporting Explicit Chain-of-Thought (CoT) Parsing
    """
    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name

    def generate(self, prompt: str, use_cot: bool = True) -> Tuple[str, str]:
        """
        Returns:
            cot_reasoning (str): The step-by-step CoT thinking path
            final_answer (str): The structured final clinical decision
        """
        # Simulated CoT reasoning generation logic
        if use_cot:
            cot_reasoning = (
                f"[{self.model_name} CoT Thought Process]:\n"
                f"1. Clinical Case Analysis: Analyzing patient symptoms and imaging features.\n"
                f"2. Subtyping Match: Assessing molecular subtyping confidence against NCCN guidelines.\n"
                f"3. Risk & Utility Assessment: Evaluating treatment intensity vs side-effects.\n"
                f"4. Conflict Resolution: Aligning recommendations with interdisciplinary evidence."
            )
        else:
            cot_reasoning = "Direct inference without CoT."

        final_answer = f"[{self.model_name} Final Recommendation based on evaluation]"
        return cot_reasoning, final_answer

    def get_token_probs(self, prompt: str, text: str) -> List[float]:
        
        return [0.92, 0.88, 0.95, 0.89]



class VectorDBRetriever:
    """
    Explicit RAG Module simulating FAISS / Milvus Vector Search for Clinical Evidence & Exemplars
    """
    def __init__(self):
        # Simulated Knowledge Base (Clinical Guidelines & Historical Exemplars)
        self.guideline_db = [
            {"id": "G1", "content": "NCCN Guideline 2026: Luminal B high-risk cases require neoadjuvant chemotherapy combined with targeted therapy.", "vec": bert_embedder.get_embedding("Luminal B neoadjuvant")},
            {"id": "G2", "content": "CSCO Guideline 2026: For low-risk early breast cancer, lumpectomy followed by endocrine therapy is recommended.", "vec": bert_embedder.get_embedding("low-risk early lumpectomy")}
        ]
        self.exemplar_db = [
            {"clinical": "55yo F, ER+, PR+, HER2-", "subtyping": "Luminal A", "mri": "BI-RADS 3", "decision": "Low-risk endocrine therapy", "vec": bert_embedder.get_embedding("Luminal A low risk")},
            {"clinical": "48yo F, ER-, PR-, HER2+", "subtyping": "HER2-enriched", "mri": "BI-RADS 5", "decision": "High-risk anti-HER2 targeted therapy + chemotherapy", "vec": bert_embedder.get_embedding("HER2 high risk")}
        ]

    def retrieve_guidelines(self, query_str: str, top_k: int = 1) -> List[str]:
        """
        RAG Retrieval Step 1: Semantic Vector Search for Guidelines
        """
        query_vec = bert_embedder.get_embedding(query_str)
        scored = []
        for doc in self.guideline_db:
            score = np.dot(query_vec, doc["vec"])
            scored.append((score, doc["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def retrieve_exemplars(self, query_str: str, top_k: int = 1) -> List[Dict[str, str]]:
        """
        RAG Retrieval Step 2: Semantic Similarity Search for Few-Shot Exemplars
        """
        query_vec = bert_embedder.get_embedding(query_str)
        scored = []
        for ex in self.exemplar_db:
            score = np.dot(query_vec, ex["vec"])
            scored.append((score, ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]


class RiskTriager:
    def __init__(self, llm: LLMInterface, rag_retriever: VectorDBRetriever):
        self.llm = llm
        self.rag = rag_retriever

    def construct_rag_input_prompt(self, case_data: Dict[str, Any]) -> str:
        """
        Formula (4.1) implementation integrated with explicit RAG Retrieval Context
        """
        raw_query = f"{case_data.get('clinical', '')} {case_data.get('subtyping', '')} {case_data.get('mri', '')}"
        
       
        retrieved_guides = self.rag.retrieve_guidelines(raw_query, top_k=1)
        retrieved_exemplars = self.rag.retrieve_exemplars(raw_query, top_k=1)

        exemplar_str = ""
        for i, ex in enumerate(retrieved_exemplars, 1):
            exemplar_str += f"\n[RAG Exemplar {i}]:\n- Clinical: {ex['clinical']}\n- Subtyping: {ex['subtyping']}\n- MRI: {ex['mri']}\n- Reference Decision: {ex['decision']}\n"

       
        prompt = f"""You are the LLM Triager.

[Retrieved Clinical Context (RAG)]:
- Augmented Guideline Evidence: {retrieved_guides[0] if retrieved_guides else 'N/A'}
- Relevant Few-Shot Exemplars:
{exemplar_str}

[Patient Profile]:
- Clinical Case Information: {case_data.get('clinical', '')}
- Molecular Subtyping Prediction: {case_data.get('subtyping', '')}
- MRI Biomarkers: {case_data.get('mri', '')}
- Socioeconomic Status: {case_data.get('family', '')}

[CoT Instructions]:
Please think step-by-step before making your risk classification:
<think>
1. Evaluate patient baseline risk based on clinical case and MRI report.
2. Cross-reference subtyping with retrieved guidelines and exemplars.
3. Balance financial limitations with treatment intensity.
</think>

Output Options:
- "Risk Level: LRT" (Low cost, baseline efficacy, mild side effects)
- "Risk Level: MRT" (Moderate cost, balanced trade-off)
- "Risk Level: HRT" (High cost, maximal efficacy, intensive supervision)

Provide your step-by-step reasoning and final decision.
"""
        return prompt

    def triage(self, case_data: Dict[str, Any]) -> Tuple[str, str, str]:
        prompt = self.construct_rag_input_prompt(case_data)
        cot_reasoning, answer = self.llm.generate(prompt, use_cot=True)
        
        if "HRT" in answer:
            risk_level = "HRT"
        elif "MRT" in answer:
            risk_level = "MRT"
        else:
            risk_level = "LRT"
            
        return risk_level, cot_reasoning, prompt



class LowRiskTeam:
    def __init__(self, llm: LLMInterface):
        self.llm = llm

    def process(self, input_prompt: str, style_prompt: str) -> str:
        final_prompt = f"{input_prompt}\n\n[CoT Instruction]: Think step-by-step to structure the low-risk treatment plan.\nFormatting Requirements:\n{style_prompt}"
        cot, answer = self.llm.generate(final_prompt, use_cot=True)
        return f"[CoT Thinking Process]:\n{cot}\n\n[Final Plan]:\n{answer}"



class MidRiskTeam:
    def __init__(self, llms: Dict[str, LLMInterface]):
        self.roles = ['Imaging', 'Pathology', 'Surgery', 'Oncology', 'Rehabilitation']
        self.llms = llms
        self.R, self.T, self.P, self.S = 3.0, 5.0, 1.0, 0.0
        self.gamma = 0.8

    def _get_payoff(self, action_i: str, action_j: str) -> Tuple[float, float]:
        if action_i == 'C' and action_j == 'C': return self.R, self.R
        elif action_i == 'C' and action_j == 'D': return self.S, self.T
        elif action_i == 'D' and action_j == 'C': return self.T, self.S
        else: return self.P, self.P

    def calculate_scc(self, vectors: List[np.ndarray]) -> float:
        v_mean = np.mean(vectors, axis=0)
        dist_sq = [np.linalg.norm(v - v_mean)**2 for v in vectors]
        return 1.0 / (1.0 + np.mean(dist_sq))

    def process(self, input_prompt: str, templates: List[str]) -> str:
        reputation = {role: 1.0 for role in self.roles}
        history_opinions = {role: "" for role in self.roles}
        history_cots = {role: "" for role in self.roles}
        
        # 1. Generate initial opinions with CoT
        for role in self.roles:
            prompt = f"You are the {role} specialist.\nData Context:\n{input_prompt}\n[CoT Requirement]: Think step-by-step regarding domain-specific risks before giving recommendations."
            cot, answer = self.llms[role].generate(prompt, use_cot=True)
            history_cots[role] = cot
            history_opinions[role] = answer

        max_rounds = 3
        for r in range(max_rounds):
            round_actions = {}
            for role in self.roles:
                opinion = history_opinions[role]
                round_actions[role] = 'C' if any(kw in opinion.lower() for kw in ["refer", "adjust", "cooperate", "align"]) else 'D'

            # Update reputation scores
            for i, r_i in enumerate(self.roles):
                payoff_sum = sum(self._get_payoff(round_actions[r_i], round_actions[r_j])[0] for j, r_j in enumerate(self.roles) if i != j)
                reputation[r_i] += self.gamma * payoff_sum

            # SCC convergence check 
            vectors = [bert_embedder.get_embedding(history_opinions[role]) for role in self.roles]
            scc = self.calculate_scc(vectors)
            if scc > 0.95:
                break

            # Multi-turn interaction with CoT reasoning update
            for role in self.roles:
                context = "\n".join([f"{k}: {v}" for k, v in history_opinions.items() if k != role])
                prompt = f"You are {role} (Reputation Score: {reputation[role]:.2f}). Peers' opinions:\n{context}\n[CoT Requirement]: Analyze peers' arguments step-by-step and adjust your plan."
                cot, answer = self.llms[role].generate(prompt, use_cot=True)
                history_cots[role] = cot
                history_opinions[role] = answer

        # Decision aggregation
        expert_vecs = [bert_embedder.get_embedding(history_opinions[role]) for role in self.roles]
        template_vecs = [bert_embedder.get_embedding(tpl) for tpl in templates]

        scores_d = np.zeros(len(templates))
        total_rep = sum(reputation.values())

        for i, role in enumerate(self.roles):
            sims = [np.dot(expert_vecs[i], t_v) for t_v in template_vecs]
            e_sims = np.exp(sims - np.max(sims))
            P_i = e_sims / e_sims.sum()
            scores_d += (reputation[role] / total_rep) * P_i

        selected_template = templates[np.argmax(scores_d)]
        
        # Build response including explicit CoT trace
        cot_summary = "\n".join([f"[{role} CoT]: {history_cots[role]}" for role in self.roles])
        return f"【MRT Final Selected Direction】: {selected_template}\n\n【Interdisciplinary CoT Reasoning Traces】:\n{cot_summary}"



class HighRiskTeam:
    def __init__(self, llms: Dict[str, LLMInterface], rag_retriever: VectorDBRetriever, theta: float = 0.85):
        self.llms = llms
        self.rag = rag_retriever
        self.theta = theta
        self.G_stag = 5.0
        self.G_hare = 2.0

    def calculate_confidence(self, prompt: str, text: str, role: str) -> float:
        probs = self.llms[role].get_token_probs(prompt, text)
        log_probs = [math.log(p) for p in probs]
        return math.exp(sum(log_probs) / len(log_probs))

    def process(self, input_prompt: str) -> str:
        # Phase 1: Stag Scouting Team (SST)
        sst_roles = ['Imaging', 'Pathology', 'Surgery']
        sst_outputs = {}
        sst_cots = {}
        confidences = {}

        for role in sst_roles:
            prompt = f"You are SST {role} specialist.\nContext:\n{input_prompt}\n[CoT Instruction]: Step-by-step verify lesion boundary, subtyping confidence, and surgical feasibility."
            cot, text = self.llms[role].generate(prompt, use_cot=True)
            sst_outputs[role] = text
            sst_cots[role] = cot
            confidences[role] = self.calculate_confidence(prompt, text, role)

       
        prod_c = math.prod(confidences.values())
        eu_stag = self.G_stag * prod_c

        if eu_stag < self.G_hare or any(c < self.theta for c in confidences.values()):
            return "【HRT High Risk / Low Confidence Threshold Triggered】: Falling back to Conservative Hare Strategy."

        # Phase 2: Stag Encirclement Team (SET) with RAG & Bottleneck Resolution
        set_roles = ['Oncology', 'Surgery', 'Radiotherapy']
        set_outputs = {}
        
        # Additional RAG lookup for complex HRT intervention protocols
        hrt_rag_evidence = self.rag.retrieve_guidelines("High risk neoadjuvant targeted protocol", top_k=1)

        for role in set_roles:
            prompt = (
                f"You are SET {role} specialist.\n"
                f"SST Diagnostic Findings:\n{sst_outputs}\n"
                f"RAG Retrieved Protocol: {hrt_rag_evidence[0] if hrt_rag_evidence else 'N/A'}\n"
                f"[CoT Instruction]: Step-by-step formulate an aggressive, synergistic curative strategy."
            )
            _, text = self.llms[role].generate(prompt, use_cot=True)
            set_outputs[role] = text

        
        max_iters = 3
        for it in range(max_iters):
            vecs = {role: bert_embedder.get_embedding(set_outputs[role]) for role in set_roles}
            roles_list = list(set_roles)
            matrix = np.zeros((len(roles_list), len(roles_list)))
            for i, r1 in enumerate(roles_list):
                for j, r2 in enumerate(roles_list):
                    matrix[i, j] = np.dot(vecs[r1], vecs[r2])

            min_val = 1.0
            min_pair = (0, 0)
            for i in range(len(roles_list)):
                for j in range(len(roles_list)):
                    if i != j and matrix[i, j] < min_val:
                        min_val = matrix[i, j]
                        min_pair = (i, j)

            if min_val >= self.theta:
                break

            r_low, r_high = roles_list[min_pair[0]], roles_list[min_pair[1]]
            feedback_prompt = (
                f"Adjust strategy based on higher-confidence peer 【{r_high}】:\n{set_outputs[r_high]}\n"
                f"[CoT Requirement]: Think step-by-step to align surgical/radiotherapy timing with systemic therapy."
            )
            _, set_outputs[r_low] = self.llms[r_low].generate(feedback_prompt, use_cot=True)

        # Phase 3: Stag Deciding Team (SDT)
        sdt_prompt = f"Senior Oncology Board Decision based on synchronized SET plans:\n{set_outputs}\n[CoT Instruction]: Synthesize all pathways step-by-step into a final curative treatment roadmap."
        sdt_cot, final_decision = self.llms['Oncology'].generate(sdt_prompt, use_cot=True)

        return (
            f"【HRT Final Pareto-Optimal Curative Plan (Stag Strategy)】:\n{final_decision}\n\n"
            f"【Final Synthesis CoT Chain】:\n{sdt_cot}"
        )




class GBDAgentsPipeline:
    def __init__(self):
        self.rag_retriever = VectorDBRetriever()
        self.default_llm = LLMInterface()
        self.triager = RiskTriager(self.default_llm, self.rag_retriever)
        self.lrt = LowRiskTeam(self.default_llm)
        
        expert_llms = {
            'Imaging': LLMInterface("Qwen3-Imaging"),
            'Pathology': LLMInterface("DeepSeek-Pathology"),
            'Surgery': LLMInterface("ChatGPT-Surgery"),
            'Oncology': LLMInterface("ChatGPT-Oncology"),
            'Rehabilitation': LLMInterface("Qwen3-Rehab"),
            'Radiotherapy': LLMInterface("Qwen3-Radio")
        }
        
        self.mrt = MidRiskTeam(expert_llms)
        self.hrt = HighRiskTeam(expert_llms, self.rag_retriever, theta=0.85)

    def run(self, case_data: Dict[str, Any]) -> str:
        # Step 1: Triaging with RAG and CoT
        risk_level, triage_cot, input_prompt = self.triager.triage(case_data)
        print(f"[System Log] Case Triage Result: {risk_level}")
        print(f"[System Log] Triage CoT Trace:\n{triage_cot}\n")

        # Step 2: Route to corresponding Game-Theoretic Team
        if risk_level == "LRT":
            style_prompt = "1. Diagnostic Summary; 2. Follow-up Advice."
            return self.lrt.process(input_prompt, style_prompt)

        elif risk_level == "MRT":
            candidate_templates = [
                "Option A: Partial Mastectomy + Periodic Chemotherapy + Close Follow-up",
                "Option B: Breast-Conserving Surgery + Radiotherapy + Adjuvant Immunotherapy",
                "Option C: Total Mastectomy + Targeted Therapy"
            ]
            return self.mrt.process(input_prompt, candidate_templates)

        elif risk_level == "HRT":
            return self.hrt.process(input_prompt)


