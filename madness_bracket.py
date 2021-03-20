from typing import Tuple, List, Dict
import re
import numpy as np
import queue


##### HELPER FUNCTIONS ########


def num_games_in_rd(rd: int, num_teams: int):
    num_rds = int(np.log2(num_teams))
    assert 0 < rd <= num_rds, "round is outside the allowable range"
    return int(num_teams / (2 ** rd))


def get_seed(team: str) -> int:
    """Returns the seeding of the team from the team name

    Args:
        team (str): team name (e.g. MW_3)

    Returns:
        int: the seed (e.g. MW_3 -> 3)
    """
    assert "_" in team
    split_ind = team.find("_")
    seed = team[split_ind + 1 :]
    return int(seed)


def get_teams(divisions: List[str], num_seeds: int) -> Dict[str, List]:
    """generates a lookup table for team names separated by division

    Example entry: {'MW': [MW_1, MW_2, ...., MW_16],
                    'E':  [E_1, E_2, ...., E_16] }

    Args:
        divisions (List[str]): List of the divisions (e.g. ['W', 'E', 'MW', 'S'])
        num_seeds (int): number of seeds in each division (e.g. march madness ==
            16)


    Returns:
        dict[str, List]: lookup table of form {'division1': [team1_name,
            team2_name, ..., etc], 'division2': [team1_name,
            team2_name, ..., etc],}
    """

    team_nums = [str(x) for x in range(1, num_seeds + 1)]
    teams = {x: [] for x in divisions}
    for div in divisions:
        for num in team_nums:
            teams[div].append(div + "_" + num)
    return teams


def decide_points_advantage(t1: str, t2: str, rd: int, win_probs):
    """one off function for making single game guesses

    Args:
        t1 (str): [description]
        t2 (str): [description]
        rd (int): [description]
        win_probs ([type]): [description]
    """
    expected_1, expected_2 = get_expected_pts(t1, t2, rd, win_probs)
    if expected_1 > expected_2:
        print(f"Choose {t1}")
    elif expected_1 < expected_2:
        print(f"Choose {t2}")
    else:
        print("Tie")
    print()


def get_expected_pts(t1: str, t2: str, rd: int, win_probs) -> Tuple[float, float]:
    """Returns a tuple of the expected points if either team was chosen as a
    winner in the given game as determined by the params passed in.

    Args:
        t1 (str): the name of team 1
        t2 (str): the name of team 2
        rd (int): the round they are matching up in
        win_probs (Dict[List[float]]):  a lookup table containing the
            probabilities each team will make it to each round

    Returns:
        Tuple[float, float]: the ordered expected points that will be earned if
            each team is selected as the winner
    """

    def get_pts(t1: str, t2: str, rd: int) -> Tuple[int, int]:
        """Determines the points that would be won for each team if the winner
        was correctly guessed

        Args:
            t1 (str): team 1 name
            t2 (str): team 2 name
            rd (int): the round they are matching up in

        Returns:
            Tuple[int, int]: the points won if correctly guessing each teams win
        """
        assert "_" in t1
        assert "_" in t2
        assert rd > 0
        seed1 = get_seed(t1)
        seed2 = get_seed(t2)

        t1_pts = rd
        t2_pts = rd
        if seed1 > seed2:
            t1_pts *= 2
            seed_diff = seed1 - seed2
            assert seed_diff > 0
            t1_pts += seed_diff
        elif seed2 > seed1:
            t2_pts *= 2
            seed_diff = seed2 - seed1
            assert seed_diff > 0
            t2_pts += seed_diff

        return (t1_pts, t2_pts)

    def get_win_prob(t1: str, t2: str, rd: int, win_probs) -> Tuple[float, float]:
        t1_prob = win_probs[t1][rd - 1]
        t2_prob = win_probs[t2][rd - 1]
        prob_sum = t1_prob + t2_prob
        return (t1_prob / prob_sum, t2_prob / prob_sum)

    win_probs = get_win_prob(t1, t2, rd, win_probs)
    pts = get_pts(t1, t2, rd)
    expected_1 = win_probs[0] * pts[0]
    expected_2 = win_probs[1] * pts[1]
    return (expected_1, expected_2)


def make_win_prob_table() -> Dict[str, List[float]]:
    """Returns a table of the probabilities that each team will make it to a
    certain round. The table is keyed first by the team name (e.g. 'MW_3' and
    the entry corresponding to that key is a list where each index is the
    probability that the team will make it to the next round)

    Example: {'MW_3': [0.75, 0.5, 0.3, 0.1]} means that team 'MW_3' has a
    predicted 75% chance of making it to round 2 and a predicted 50% chance of
    making it to round 3
    """

    def parse_win_probs(win_prob_str: str) -> Dict[str, List[float]]:
        """Parses a string of known format to build a lookup probability table
        from it. This assumes that the team names (e.g. 'MW_3') are separators
        for the probabilities inside the string.

        The expected format is (team_1 # # # # team_2 # # # # team_3 # # # #)
        where the '#' indicate a probability value associated with the previous team


        Args:
            win_prob_str (str): the string to parse

        Returns:
            dict[str, List[float]]: the lookup table for probabilities
        """

        def is_seed(entry: str) -> bool:
            """returns whether the string passed in is a team/seed value (e.g.
            'MW_3' -> True, 0.0075 -> False)

            Args:
                entry (str): the entry to check

            Returns:
                bool: if entry indicates a new team
            """
            regexp = re.compile(r"[SEMW]")
            if regexp.search(entry):
                return True
            else:
                return False

        def sanitize_prob_entry(entry: str) -> float:
            if "<" in entry:
                return 0.0
            else:
                return float(entry)

        def sanitize_seed_entry(entry: str) -> str:
            regexp = re.compile(r"[0-9]{1,2}")
            res = regexp.search(entry)
            seed = res.group(0)
            div = entry[len(seed) :]
            return str(div + "_" + seed)

        entries = win_prob_str.split()
        win_probs = {}
        last_seed = None
        for e in entries:
            # if seed start adding info for this seed
            if is_seed(e):
                last_seed = sanitize_seed_entry(e)
                win_probs[last_seed] = []
            else:
                # we have already started parsing probs for this seed
                entry = sanitize_prob_entry(e)
                assert (
                    entry <= 100.0
                ), "Error, win probability greater than 100 percent observed"
                win_probs[last_seed].append(entry)

        return win_probs

    win_prob_str = "1W  99.6   91.2   77.3   60.5   46.3   34.4   1MW 96.1   67.8   52.7   34.5   22.2   10.7   1E  98.4   72.7   53.3   37.6   17.4   10.4   1S  97.5   67.2   46.6   31.5   18.0    8.2   2MW 94.4   70.6   49.9   28.2   16.8    7.4   2W  91.0   70.6   49.7   19.3   11.3    6.4   2S  93.8   66.2   41.9   21.0   10.1    3.7   2E  95.7   58.6   37.9   17.8    5.8    2.7   5S  80.7   45.8   18.4    9.5    4.0    1.2   8MW 61.9   22.3   13.9    6.9    3.3    1.1   4S  72.6   40.1   15.7    8.0    3.3    1.0   4E  81.2   46.3   16.8    8.4    2.3    0.9   4W  74.3   43.3   10.3    4.7    2.1    0.9   3S  78.1   43.8   21.0    8.5    3.2    0.9   5MW 73.9   45.3   15.4    6.7    2.8    0.8   6W  70.3   41.0   17.5    4.5    1.9    0.8   9S  56.4   20.1   10.8    5.6    2.4    0.7   5E  68.0   37.2   13.4    6.6    1.8    0.7   6MW 62.2   35.7   14.7    5.8    2.4    0.7   6S  60.5   32.8   15.4    6.2    2.3    0.6   7E  57.4   25.3   14.3    5.8    1.6    0.6   3MW 84.1   44.1   17.0    6.3    2.5    0.6   3W  82.4   43.9   17.5    4.1    1.6    0.6   6E  63.9   36.5   16.2    6.1    1.6    0.6   3E  74.8   40.5   17.5    6.5    1.6    0.6   5W  70.9   38.3    7.3    3.1    1.3    0.5   9E  53.1   15.4    7.8    3.7    0.9    0.3   4MW 73.6   37.6   10.6    4.0    1.4    0.3   8S  43.6   12.5    6.0    2.7    1.0    0.2   10E  42.6   15.6    7.7    2.6    0.6    0.2   8E  46.9   11.8    5.7    2.5    0.6    0.2   10MW 54.6   16.2    7.6    2.5    0.9    0.2   7S  55.6   19.3    8.6    2.8    0.8    0.2   9MW 38.1    9.4    4.8    1.8    0.6    0.1   7W  54.4   15.6    6.9    1.3    0.4    0.1   11S  39.5   17.1    6.2    1.9    0.5    0.1   7MW 45.4   12.2    5.2    1.5    0.5    0.10  11MW 37.8   17.2    5.2    1.5    0.5    0.08  10S  44.4   13.5    5.3    1.5    0.4    0.07  8W  53.6    5.1    2.2    0.7    0.2    0.06  10W  45.6   11.3    4.5    0.7    0.2    0.05  11E  20.9    9.5    3.1    0.9    0.2    0.04  9W  46.4    3.6    1.4    0.4    0.1    0.03  12E  32.0   11.7    2.4    0.7    0.1    0.03  11W  18.6    7.6    2.1    0.3    0.08   0.02  13S  27.4    9.0    1.8    0.5    0.10   0.01  11E  15.2    6.1    1.7    0.4    0.06   0.01  12W  29.1   10.2    0.9    0.2    0.05   0.010 12MW 26.1    9.4    1.4    0.3    0.06   0.008 13W  25.7    8.2    0.6    0.1    0.03   0.005 14E  25.2    7.4    1.5    0.2    0.02   0.004 14S  21.9    6.3    1.4    0.3    0.04   0.004 13MW 26.4    7.7    1.0    0.2    0.04   0.004 11W  11.0    3.8    0.8    0.08   0.02   0.003 13E  18.8    4.7    0.6    0.1    0.010  0.001 12S  19.3    5.0    0.6    0.1    0.02   <.001 14W  17.6    3.7    0.5    0.03   0.004  <.001 15W  9.0    2.5    0.6    0.04   0.005  <.001 14MW 15.9    2.9    0.4    0.04   0.005  <.001 15MW 5.6    1.0    0.1    0.01   <.001  <.001 15S  6.2    1.0    0.1    0.009  <.001  <.001 16MW 3.9    0.5    0.09   0.009  <.001  <.001 15E  4.3    0.4    0.05   0.003  <.001  <.001 16S  2.5    0.2    0.02   0.001  <.001  <.001 16E  1.2    0.09   0.007  <.001  <.001  <.001 16W  0.2    0.02   0.002  <.001  <.001  <.001 16E  0.5    0.02   0.001  <.001  <.001  <.001 16W  0.2    0.02   0.001  <.001  <.001  <.001"
    win_probs = parse_win_probs(win_prob_str)
    return win_probs



class Tournament:
    class Game:
        def __init__(
            self,
            t1: str = None,
            t2: str = None,
            child_game_1=None,
            child_game_2=None,
            winner: int = None,
            rd: int = None,
        ):
            assert winner in [0, 1, None]
            assert rd > 0
            self.t1 = t1
            self.t2 = t2
            self.rd = rd
            self.winner = winner
            self.child1 = child_game_1
            self.child2 = child_game_2
            self.win_probs = make_win_prob_table()

            # all points from all children games (recursively summed to leaves of
            # tree)
            self.pts_summed = 0

        def __str__(self):
            if self.winner == 0:
                return f"Matchup {self.t1} vs {self.t2} Rd {self.rd} Winner {self.t1}: Expected Pts = {self.pts_summed}"
            elif self.winner == 1:
                return f"Matchup {self.t1} vs {self.t2} Rd {self.rd} Winner {self.t2}: Expected Pts = {self.pts_summed}"
            elif self.winner is None:
                return f"Matchup {self.t1} vs {self.t2} Rd {self.rd} Winner {self.winner}: Expected Pts = {self.pts_summed}"

        def get_winning_team(self) -> str:
            """Gets name of winning team in this game

            Returns:
                str: name of the winning team for this game
            """
            assert (
                self.t1 is not None
            ), "tried to find winning team but no team initialized yet"
            assert (
                self.t2 is not None
            ), "tried to find winning team but no team initialized yet"

            if self.winner == 0:
                return self.t1
            elif self.winner == 1:
                return self.t2

        def fill_team_from_child_winners(self) -> None:
            """updates the teams for this game from the winners of the child
            games.

            NOTE: To avoid repeat computations, DOES NOT recursively call this
            function for children games.
            """
            if self.child1 is None and self.child2 is None:
                return

            assert self.child1 is not None and self.child2 is not None

            self.t1 = self.child1.get_winning_team()
            self.t2 = self.child2.get_winning_team()

            assert self.t1 != self.t2

        def update_pts_summed_recursively(self) -> None:
            """updates the member object pts_summed to be the sum of the
            expected number of points for this game plus all of the points from
            its descendent games
            """
            assert self.t1 is not None and self.t2 is not None

            # points from this game
            expected_pts = get_expected_pts(self.t1, self.t2, self.rd, self.win_probs)[
                self.winner
            ]

            # check if leaf game
            if self.child1 is None or self.child2 is None:
                assert self.child1 is None and self.child2 is None
                self.pts_summed = expected_pts
            else:
                self.child1.update_pts_summed_recursively()
                self.child2.update_pts_summed_recursively()
                self.pts_summed = (
                    expected_pts + self.child1.pts_summed + self.child2.pts_summed
                )

        def check_consistency(self) -> None:
            """Checks consistency for this game and recursively checks the
            consistency for all of the descendent games. Consistency is
            determined by whether the teams in this game are the winners of the
            children games.
            """
            if self.child1 is None and self.child2 is None:
                return

            assert self.t1 == self.child1.get_winning_team()
            assert self.t2 == self.child2.get_winning_team()

            self.child1.check_consistency()
            self.child2.check_consistency()

        def get_children(self) -> Tuple:
            """Returns children games of this game

            Returns:
                Tuple[Game]: the children games to this game. Returns an empty
                    tuple if there are none
            """
            if self.child1 is None or self.child2 is None:
                assert (
                    self.child1 is None and self.child2 is None
                ), "Error: only one child game is none"

                return ()
            else:
                return (self.child1, self.child2)

        def get_points(self) -> float:
            """Returns the summed expected points for this game and all of it's
            descendant games. If is 0 then is possibly because  need to call
            update_pts_summed_recursively() first

            Returns:
                float: the expected number of points
            """
            return self.pts_summed

        def set_random_winner_recursively(self):
            """Generates (mostly) random guess on the winner. There are special
            bracket conditions as determined by function
            is_special_bracket_condition where certain win decisions are
            imposed, thus eliminating some randomness.

            """

            def is_special_bracket_condition(team, rd):
                """Just a function for special edge cases I want to have to enforce
                my preferences (e.g. Gonzaga must wins game in first 4 rds)

                Args:
                    team (str): the team string (e.g. 'MW_3')
                    rd (int): the round number

                Returns:
                    bool: whether this team will be determined to win
                """

                if rd == 1:
                    if get_seed(team) <= 3:
                        return True

                if team == "W_1" and rd < 5:
                    return True

                return False

            # if have children determine their winners first (recursion)
            if self.child1 is not None and self.child2 is not None:
                self.child1.set_random_winner_recursively()
                self.child2.set_random_winner_recursively()
                self.fill_team_from_child_winners()

            # first check for predetermined winner conditions. If none are met
            # then randomly assign the winner
            if is_special_bracket_condition(self.t1, self.rd):
                self.winner = 0
            elif is_special_bracket_condition(self.t2, self.rd):
                self.winner = 1
            else:
                guess = round(np.random.uniform())
                self.winner = guess

    def __init__(self):
        self.regions = [x.upper() for x in ["mw", "w", "s", "e"]]
        self.n_seeds = 16
        self.num_teams = len(self.regions) * self.n_seeds
        self.num_rds = int(np.log2(self.num_teams))
        self.games = [[] for x in range(self.num_rds)]

        # initialize first round
        teams = get_teams(self.regions, self.n_seeds)
        curr_round_ind = 0
        for region in self.regions:
            for i in range(0, int(self.n_seeds / 2)):
                t1_ind = i + 1
                t2_ind = self.n_seeds - i
                t1 = teams[region][t1_ind - 1]
                t2 = teams[region][t2_ind - 1]
                if i < 3:
                    game = self.Game(t1, t2, winner=0, rd=curr_round_ind + 1)
                    self.games[curr_round_ind].append(game)
                else:
                    game = self.Game(t1, t2, rd=curr_round_ind + 1)
                    self.games[curr_round_ind].append(game)

        for round_ind in range(1, self.num_rds):
            num_games = num_games_in_rd(round_ind + 1, self.num_teams)
            for game_ind in range(num_games):
                child_game_1 = self.games[round_ind - 1][game_ind * 2]
                child_game_2 = self.games[round_ind - 1][game_ind * 2 + 1]
                game = self.Game(
                    child_game_1=child_game_1,
                    child_game_2=child_game_2,
                    rd=round_ind + 1,
                )
                self.games[round_ind].append(game)

        assert len(self.games[-1]) == 1, "incorrectly accessing the title game"
        title_game = self.games[-1][0]
        self.title_game = self.games[-1][0]

    def print_info(self):
        """Start at championship game and recursively work down tree in BFS
        order to print status of the games. Does this via FIFO queue
        """
        title_game = self.title_game

        print(
            "\n************************\nPRINTING BRACKET INFO\n************************"
        )
        game_queue = queue.Queue()
        game_queue.put(title_game)
        curr_round = self.num_rds
        while not game_queue.empty():
            game = game_queue.get()
            if game.rd != curr_round:
                curr_round = game.rd
                print()
            print(game)

            children = game.get_children()
            assert len(children) in [
                0,
                2,
            ], f"Check the tree construction, number of children was {len(children)} but must be either 0 or 2"

            if len(children) == 2:
                child1 = children[0]
                child2 = children[1]
                game_queue.put(child1)
                game_queue.put(child2)

        print(f"\n Total Expected Pts: {title_game.pts_summed}\n")
        print("************************")
        print("\n")

    def test_random_assignment(self):
        self.title_game.set_random_winner_recursively()
        self.title_game.update_pts_summed_recursively()
        points = self.title_game.get_points()
        return points


if __name__ == "__main__":
    """This is a file meant to try to predict a successful March Madness
    bracket. There are 2 key objects defined in this file: Tournament and
    Game. Each tournament is March Madness styled and is a collection of
    games. Beginning with the first round, winners for each game are
    determined by some measure and then the games in the second round look at
    their 'children' games and take the winners of each of their 2 children
    games to be the participants in their own game.
    `
    The tournament is a tree style object, with the root being the
    championship game and each node having a branching factor of 2.

    An example of how to generate a tournament instance (determine winners of
    games and update the whole tree) can be seen in test_random_assignment()
    """

    # TODO make deterministic way of iterating through possible combinations
    # TODO evaluate other ways of determining the expected value of each bracket
    # based on the consideration that incorrectly guessing a game eliminates the
    # probability of winning points down the road with that team

    tournament = Tournament()
    max_pts = 0
    # num_trials = 9999999
    # for i in range(num_trials):
    i = 0
    while True:
        i += 1
        pts = tournament.test_random_assignment()
        if pts > max_pts:
            max_pts = pts
            tournament.print_info()

        if i % 10000 == 0:
            print(f"Random Trial: {i}")