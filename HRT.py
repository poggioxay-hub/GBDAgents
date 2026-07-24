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

        # Confidence & EU Check 
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

        # Bottleneck Optimization Loop
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


# Algorithm 2: HRT Risk-Sensitive Stag Hunt Coordination

# Input: Patient evidence P_input; HRT Agents N = {1, ..., K}; Risk threshold theta = 0.85; Max iterations I_max = 3.
# Output: High-risk coordinated strategy D_Stag OR Conservative fallback D_Hare.

# /* Step 1: Confidence Assessment and Expected Utility Check */
for i in N:
    T[i], E[i] = generate_diagnosis_and_extract_tokens(agent=i, P_input=P_input) # Generate diagnosis T_i and extract key clinical tokens E_i;
    C[i] = exp((1 / len(E[i])) * sum(log(P(w | P_input)) for w in E[i]))           # Compute token confidence C_i <- exp( (1 / |E_i|) * sum_{w in E_i} log P(w | P_input) );

for i in N:
    EU_Stag[i] = G_Stag * prod(C[j] for j in N if j != i)                        # EU_i(Stag) <- G_Stag * prod_{j != i} C_j; // G_Stag > G_Hare
    if C[i] < theta or EU_Stag[i] < G_Hare:
        return D_Hare                                                            # return Conservative Fallback Plan D_Hare; // Safety trigger

# /* Step 2: Synergy Bottleneck Identification and Dynamic Alignment */
t = 1
while t <= I_max:
    v_t = {i: Encoder(T[i]) for i in N}                                          # Compute embeddings v_i^(t) = Encoder(T_i^(t)), for all i in N;
    M_t = [[CosSim(v_t[i], v_t[j]) for j in N] for i in N]                       # Construct Synergy Matrix M^(t) in R^(K x K), where M_ij^(t) = CosSim(v_i^(t), v_j^(t));
    S_synergy_t = min(M_t[j][k] for j in N for k in N if j != k)                 # Find global bottleneck score: S_synergy^(t) <- min_{j != k} M_jk^(t);
    
    if S_synergy_t >= theta:
        break                                                                    # break; // Global semantic alignment achieved
        
    j_star, k_star = argmin_pair(M_t, condition=lambda j, k: C[k] > C[j])        # Locate weakest pair (j*, k*) = arg min_{j != k} M_jk^(t) with C_k* > C_j*;
    P_constraint = Format(T[k_star], Medical_Guidelines)                         # P_constraint <- Format(T_k*^(t), Medical Guidelines);
    T[j_star] = LLM[j_star](T[j_star], P_constraint)                            # T_j*^(t+1) ~ LLM_j*(T_j*^(t), P_constraint); // Dynamic context injection
    t = t + 1

# /* Step 3: Final Decision Synthesis */
D_Stag = SDT_Synthesizer([T[i] for i in N])                                      # D_Stag <- SDT_Synthesizer({T_i^(t)}_{i=1}^K);
return D_Stag                                                                    # return D_Stag;