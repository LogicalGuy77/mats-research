"""References for the semester report (key, formatted entry). Checked against the arXiv / proceedings / blog pages."""

REFS = [
    ("AbuBaker25", "M. Abu Baker, L. Babu-Saheer. Mechanistic Exploration of Backdoored Large Language Model "
                   "Attention Patterns. arXiv:2508.15847, 2025."),
    ("Antoun26", "W. Antoun, F. Kulumba, T. Lasnier, B. Sagot, D. Seddah. LLM Forensics: Where Do Backdoors Hide? "
                 "Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders. Findings of EMNLP 2026. "
                 "arXiv:2609.07746."),
    ("Arditi24", "A. Arditi, O. Obeso, A. Syed, D. Paleka, N. Panickssery, W. Gurnee, N. Nanda. Refusal in Language "
                 "Models Is Mediated by a Single Direction. NeurIPS 2024. arXiv:2406.11717."),
    ("Arnav25", "B. Arnav, P. Bernabeu-Perez, N. Helm-Burger, T. Kostolansky, H. Whittingham, M. Phuong. CoT "
                "Red-Handed: Stress Testing Chain-of-Thought Monitoring. NeurIPS 2025. arXiv:2505.23575."),
    ("Bailey24", "L. Bailey, A. Serrano, A. Sheshadri, M. Seleznyov, J. Taylor, E. Jenner, J. Hilton, S. Casper, "
                 "C. Guestrin, S. Emmons. Obfuscated Activations Bypass LLM Latent-Space Defenses. arXiv:2412.09565, "
                 "2024."),
    ("Baker25", "B. Baker, J. Huizinga, L. Gao, Z. Dou, M. Y. Guan, A. Madry, W. Zaremba, J. Pachocki, D. Farhi. "
                "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation. OpenAI, "
                "arXiv:2503.11926, 2025."),
    ("Belrose23", "N. Belrose, I. Ostrovsky, L. McKinney, Z. Furman, L. Smith, D. Halawi, S. Biderman, J. Steinhardt. "
                  "Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112, 2023."),
    ("Boppana26", "S. Boppana, A. Ma, M. Loeffler, R. Sarfati, E. Bigelow, A. Geiger, O. Lewis, J. Merullo. Reasoning "
                  "Theater: Disentangling Model Beliefs from Chain-of-Thought. arXiv:2603.05488, 2026."),
    ("Chen25", "Y. Chen, J. Benton, A. Radhakrishnan, J. Uesato, C. Denison, J. Schulman, A. Somani, P. Hase, "
               "M. Wagner, F. Roger, V. Mikulik, S. R. Bowman, J. Leike, J. Kaplan, E. Perez. Reasoning Models Don't "
               "Always Say What They Think. Anthropic, arXiv:2505.05410, 2025."),
    ("Chua25", "J. Chua, J. Betley, M. Taylor, O. Evans. Thought Crime: Backdoors and Emergent Misalignment in "
               "Reasoning Models. arXiv:2506.13206, 2025."),
    ("ChuaEvans25", "J. Chua, O. Evans. Are DeepSeek R1 And Other Reasoning Models More Faithful? arXiv:2501.08156, "
                    "2025."),
    ("DS25", "DeepSeek-AI (D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, et al.). DeepSeek-R1: Incentivizing Reasoning "
             "Capability in LLMs via Reinforcement Learning. Nature 645, 633-638 (2025). arXiv:2501.12948."),
    ("Emmons25", "S. Emmons, E. Jenner, D. K. Elson, R. A. Saurous, S. Rajamanoharan, H. Chen, I. Shafkat, R. Shah. "
                 "When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors. Google DeepMind, "
                 "arXiv:2507.05246, 2025."),
    ("Geva21", "M. Geva, R. Schuster, J. Berant, O. Levy. Transformer Feed-Forward Layers Are Key-Value Memories. "
               "EMNLP 2021. arXiv:2012.14913."),
    ("GoldowskyDill25", "N. Goldowsky-Dill, B. Chughtai, S. Heimersheim, M. Hobbhahn. Detecting Strategic Deception "
                        "with Linear Probes. ICML 2025, PMLR 267:19755-19786. arXiv:2502.03407."),
    ("Hu21", "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen. LoRA: Low-Rank "
             "Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685."),
    ("Hubinger24", "E. Hubinger, C. Denison, J. Mu, M. Lambert, M. Tong, M. MacDiarmid, T. Lanham, D. M. Ziegler, "
                   "et al. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. "
                   "arXiv:2401.05566, 2024."),
    ("Korbak25", "T. Korbak, M. Balesni, E. Barnes, Y. Bengio, J. Benton, J. Bloom, M. Chen, A. Cooney, A. Dafoe, "
                 "A. Dragan, et al. Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety. "
                 "arXiv:2507.11473, 2025."),
    ("Lanham23", "T. Lanham, A. Chen, A. Radhakrishnan, B. Steiner, et al., S. R. Bowman, E. Perez. Measuring "
                 "Faithfulness in Chain-of-Thought Reasoning. arXiv:2307.13702, 2023."),
    ("Lasnier26", "T. Lasnier, W. Antoun, F. Kulumba, B. Sagot, D. Seddah. Language Triggers Hijack Language Circuits: "
                  "A Mechanistic Analysis of Backdoor Behaviors in Large Language Models. arXiv:2602.10382, 2026."),
    ("MacDiarmid24", "M. MacDiarmid, T. Maxwell, N. Schiefer, J. Mu, J. Kaplan, D. Duvenaud, S. Bowman, A. Tamkin, "
                     "E. Perez, M. Sharma, C. Denison, E. Hubinger. Simple probes can catch sleeper agents. Anthropic, "
                     "April 2024. https://www.anthropic.com/research/probes-catch-sleeper-agents"),
    ("Mahmoud26", "O. Mahmoud, A. M. Kassem, T. G. Karimpanal, B. L. Semage, N. Rostamzadeh, G. Farnadi, S. Rana. "
                  "Shared Latent Structures Enable Unified Backdoor Detection and Mitigation in LLMs. "
                  "arXiv:2606.07963, 2026."),
    ("Marks25", "S. Marks, J. Treutlein, T. Bricken, J. Lindsey, J. Marcus, S. Mishra-Sharma, D. Ziegler, E. Ameisen, "
                "J. Batson, et al. Auditing Language Models for Hidden Objectives. Anthropic, arXiv:2503.10965, 2025."),
    ("Meng22", "K. Meng, D. Bau, A. Andonian, Y. Belinkov. Locating and Editing Factual Associations in GPT. "
               "NeurIPS 2022. arXiv:2202.05262."),
    ("Mohammadkhani26", "M. Mohammadkhani, M. Krishna, Y. Sarrof, M. Hahn. Hidden Reasoning Must Leak, but Need Not Be "
                        "Readable: Fundamental Opportunities and Limits for Chain-of-Thought Monitoring. "
                        "arXiv:2609.37312, 2026."),
    ("nostalgebraist20", "nostalgebraist. interpreting GPT: the logit lens. LessWrong, August 2020. "
                         "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"),
    ("Qwen25", "Qwen Team (A. Yang, A. Li, B. Yang, et al.). Qwen3 Technical Report. arXiv:2505.09388, 2025."),
    ("Roger23", "F. Roger, R. Greenblatt. Preventing Language Models From Hiding Their Reasoning. arXiv:2310.18512, "
                "2023."),
    ("Severi26", "G. Severi, S. Mirza, B. Bullwinkel, A. Minnich. Evading Chain-of-Thought Monitoring Through Model "
                 "Poisoning. Microsoft, arXiv:2608.02820, 2026."),
    ("Skaf25", "J. Skaf, L. Ibanez-Lissen, R. McCarthy, C. Watts, V. Georgiv, H. Whittingham, L. Gonzalez-Manzano, "
               "D. Lindner, C. Tice, E. J. Young, P. Radmard. Large language models can learn and generalize "
               "steganographic chain-of-thought under process supervision. NeurIPS 2025. arXiv:2506.01926."),
    ("Soligo25", "A. Soligo, E. Turner, S. Rajamanoharan, N. Nanda. Convergent Linear Representations of Emergent "
                 "Misalignment. arXiv:2506.11618, 2025."),
    ("Turpin23", "M. Turpin, J. Michael, E. Perez, S. R. Bowman. Language Models Don't Always Say What They Think: "
                 "Unfaithful Explanations in Chain-of-Thought Prompting. NeurIPS 2023. arXiv:2305.04388."),
    ("Wei22", "J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. Le, D. Zhou. "
              "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022. "
              "arXiv:2201.11903."),
    ("Zhang24", "F. Zhang, N. Nanda. Towards Best Practices of Activation Patching in Language Models: Metrics and "
                "Methods. ICLR 2024. arXiv:2309.16042."),
    ("Zou23", "A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A.-K. Dombrowski, "
              "et al. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405, 2023."),
]
