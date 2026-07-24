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


# Algorithm 1: MRT Game-Theoretic Iterative Discussion and Decision Aggregation

# Input: Patient evidence P_input; Discipline agents N = {1, ..., K}; Max rounds T_max = 3; Discount factor gamma = 0.9.
# Output: Final synthesized treatment plan D_final.

# /* Phase 1: Initial Proposal Generation */
for i in N:
    T_0[i] = LLM[i](P_input)                            # Generate initial proposal T_i^(0) ~ LLM_i(P_input);
    R_0[i] = 0                                          # Initialize cumulative reputation R_i^(0) <- 0;

# /* Phase 2: Game-Theoretic Iterative Discussion */
for t in range(1, T_max + 1):
    for i in N:
        P_t[i] = construct_prompt(history={T_prev[j] for j in N if j != i})  # Construct prompt P_i^(t) with history {T_j^(t-1)}_{j != i};
        T_t[i] = LLM[i](P_t[i])                          # Generate response T_i^(t) ~ LLM_i(P_i^(t));
        a_t[i] = extract_action(T_t[i])                  # Extract strategy action a_i^(t) in {Cooperate, Defect} from T_i^(t);

    # /* Calculate pairwise payoff and update reputation */
    for i in N:
        U_t[i] = sum(Payoff(a_t[i], a_t[j]) for j in N if j != i) # U_i^(t) <- sum_{j != i} Payoff(a_i^(t), a_j^(t)); // Pairwise Prisoner's Dilemma
        R_t[i] = R_prev[i] + (gamma ** t) * U_t[i]        # R_i^(t) <- R_i^(t-1) + gamma^t * U_i^(t); // Cumulative reputation update

    # /* Check Semantic Consensus Convergence (SCC) */
    v_t = {i: Encoder(T_t[i]) for i in N}                 # Compute embeddings v_i^(t) = Encoder(T_i^(t)), for all i in N;
    S_t = Var(list(v_t.values()))                        # S^(t) <- Var({v_i^(t)}_{i=1}^K); // Semantic consensus variance
    
    if is_converged(S_t) or not new_clinical_query_raised(): # if S^(t) converged or no new clinical query raised then
        break                                            # break;

# /* Phase 3: Reputation-Weighted Decision Aggregation */
for m in M:                                              # for each candidate template m in M do
    for i in N:
        P_i[m] = Softmax(CosSim(v_T[i], v_ref[m]))       # P_i(m) <- Softmax(CosSim(v_i^(T), v_ref^(m)));

    Score[m] = sum((R_T[i] / sum(R_T[k] for k in N)) * P_i[m] for i in N) # Score(m) <- sum_{i=1}^K ( (R_i^(T) / sum_{k=1}^K R_k^(T)) * P_i(m) );

m_star = argmax(Score)                                   # m* <- argmax_m Score(m);
D_final = LLMSynthesizer(P_input, Candidate[m_star], R_T) # D_final <- LLM_Synthesizer(P_input, Candidate_{m*}, {R_i^(T)}_{i=1}^K);
return D_final                                           # return D_final;