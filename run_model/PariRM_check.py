import llm_blender
blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM") # load ranker checkpoint

inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [["get out!", "hi! I am fine, thanks!", "bye!"], 
                    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"]]
ranks = blender.rank(inputs, candidates_texts, return_scores=False, batch_size=1)
print(ranks)

scores = blender.rank(inputs, candidates_texts, return_scores=True, batch_size=1)
print(scores)
# ranks is a list of ranks where ranks[i][j] represents the ranks of candidate-j for input-i
"""
ranks -->
array([[3, 1, 2], # it means "hi! I am fine, thanks!" ranks the 1st, "bye" ranks the 2nd, and "get out!" ranks the 3rd. 
       [1, 3, 2]], # it means "I love you too"! ranks the the 1st, and "I hate you!" ranks the 3rd.
       dtype=int32) 

"""

inputs = ["hello, how are you!", "I love you!"]
candidates_texts = [["get out!", "hi! I am fine, thanks!", "bye!"], 
    ["I love you too!", "I hate you!", "Thanks! You're a good guy!"]]
rewards = blender.rank_with_ref(inputs, candidates_texts, return_scores=True, batch_size=2, mode="longest")
print("Rewards for input 1:", rewards[0]) # rewards of candidates for input 1
print('All Rewards:', rewards) # rewards of all candidates for all inputs