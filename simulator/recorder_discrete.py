# Requirements:
# - orientations (15-159° with 36° increments)
# - 15, 51, 87, 123, 159
# - Counterclockwise
# - Calibration
# - Every block should represnt the 5 orientations equally
# - Shuffle the order

from joystick_board import stickBoard

board = stickBoard()

board.setup_exp()
board.setup_game()

for sess_idx in range(board.N_SESS):
    for trial_idx in range(board.N_TRIAL):
        board.run_trial(sess_idx, trial_idx)

board.stop_game()