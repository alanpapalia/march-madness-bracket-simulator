# march-madness-bracket-simulator

This is a (very small) python script I wrote to try to bring some science into my
March Madness brackets. The current version took a long string of predicted
probabilities of each team making it to a given round from the venerable [Ken
Pomeroy](https://kenpom.com/) and parsed it into a lookup table which is used in
a pseudo-statistical manner to predict the win probability of each team at each
round. All of this happens in the function *make_win_prob_table()*.

These win probabilities are then scaled by the points won in each round
depending on which team wins. Currently the system is setup for a 'upset'-style
bracket so there are custom points rules which give additional points for
correctly guessing upsets. The round scoring happens in the function
*get_pts()*, which is nested inside of *get_expected_pts()*.

This system is V0.1 and is incredibly naive and should be improved to determine bracket
scoring strategies that both look at their historical performance and robustness
to missed guesses.
