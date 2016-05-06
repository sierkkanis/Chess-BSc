#import chess
import numpy as np
import random
import theano
import theano.tensor as T
import lasagne

board = 0
board2D = 0
pieceDict = {'K' : 0, 'k' : 1, 'P' : 2}
turn = True
halfmoves = 0
reward = 0
discount = 0.9

# reset all board parameters
def boardSetup(amountPieces):
	global board
	global turn
	global halfmoves
	global reward
	board = np.zeros(64*amountPieces).reshape((amountPieces,8,8))
	turn = True
	halfmoves = 0
	reward = 0

# set piece, doesn't remove the old
def setPiece(piece, x, y):
	global board
	global pieceDict
	board[pieceDict.get(piece)][y][x] = (pieceDict.get(piece) + 1)

def setPieceRandom(piece):
	global board
	global pieceDict
	xRandom = random.randint(0,7)
	yRandom = random.randint(0,7)
	board[pieceDict.get(piece)][yRandom][xRandom] = (pieceDict.get(piece) + 1)

def movePiece(piece, x, y):
	global board
	global pieceDict
	global turn
	global halfmoves
	for n in range(x):
		for m in range(y):
			board[pieceDict.get(piece)][m][n] = 0
	board[pieceDict.get(piece)][y][x] = (pieceDict.get(piece) + 1)
	halfmoves = halfmoves + 1
	if turn == True:
		turn = False
	else:
		turn = True

# moves whiteKing into direction
def moveKing(direction):
	global board
	global reward
	x, y = getCoor('K')
	if direction == 0 and y != 0:
		board[0][y-1][x] = 1
		board[0][y][x] = 0
	if direction == 1 and y != 0 and x != 0:
		board[0][y-1][x-1] = 1
		board[0][y][x] = 0
	if direction == 2 and y != 0 and x != 7:
		board[0][y-1][x+1] = 1
		board[0][y][x] = 0
	if direction == 3 and x != 0:
		board[0][y][x-1] = 1
		board[0][y][x] = 0
	if direction == 4 and x != 7:
		board[0][y][x+1] = 1
		board[0][y][x] = 0
	if direction == 5 and y != 7:
		board[0][y+1][x] = 1
		board[0][y][x] = 0
	if direction == 6 and y != 7 and x != 0:
		board[0][y+1][x-1] = 1
		board[0][y][x] = 0
	if direction == 7 and y != 7 and x != 7:
		board[0][y+1][x+1] = 1
		board[0][y][x] = 0
	# reward function
	xNew, yNew = getCoor('K')
	if yNew < y:
		reward += 1
		return 1
	else:
		return 0
		#moveKingRandom()
		# skips the move now

# whiteking
def moveKingRandom():
	moveKing(random.randint(0,7))

def boardTo2D():
	global board2D
	global board
	board2D = np.chararray((8,8))
	board2D[:] = '.'
	pieceDict = {0 : 'K', 1 : 'k', 2: 'P'}
	for piece in range(len(board)):
		x, y = getCoor(pieceDict.get(piece))
		board2D[y][x] = pieceDict.get(piece)

def printBoard():
	global board2D
	global turn
	global halfmoves
	global reward
	boardTo2D()
	print board2D
	print ' '
	"""
	print 'turn = '
	print turn
	#print 'position is legal = '
	#print isLegal()
	print 'halfmoves = '
	print halfmoves
	print 'terminal state '
	print isTerminalState()
	print 'reward = '
	print reward
	"""

def getCoor(piece):
	global board
	piece = pieceDict.get(piece)
	for y in range(len(board[piece])):
		for x in range(len(board[piece][y])):
			if board[piece][y][x] != 0:
				return x, y

# for black king
def inCheck():
	global board
	xP, yP = getCoor('P')
	xk, yk = getCoor('k')
	if yk == yP -1 and (xk == xP - 1 or xk == xP + 1):
		return True
	else: 
		return False

# add pieces on top of each other
def isLegal():
	global turn
	if turn == True and inCheck():
		return False
	xk, yk = getCoor('k')
	xK, yK = getCoor('K')
	if ((yk == yK -1 and (xk == xK - 1 or xk == xK or xk == xK + 1))
		or (yk == yK and (xk == xK - 1 or xk == xK or xk == xK + 1))
		or (yk == yK +1 and (xk == xK - 1 or xk == xK or xk == xK + 1))):
		return False
	else:
		return True

# if king's at the end of the board
def isTerminalState():
	global reward
	x, y = getCoor('K')
	if y == 0:
		#reward += 10
		return True
	else:
		return False
"""
boardSetup(3)
setPiece('K', 1, 1)
setPiece('k', 5, 6)
setPiece('P', 4, 5)
printBoard()
movePiece('K', 3, 3)
printBoard()
movePiece('k', 5, 4)
printBoard()
movePiece('k', 4, 4)
printBoard()
moveKing(0)
printBoard()
moveKingRandom()
printBoard()

"""
boardSetup(1)
setPieceRandom('K')
#printBoard()


"""
notes:

"""

"""
NETWORK
"""

# A very simple network, a single layer with one neuron per target class.
# Using the softmax activation function gives us a probability distribution at the output.
l_in = lasagne.layers.InputLayer((1, 64, 1))
l_out = lasagne.layers.DenseLayer(l_in, num_units=8, nonlinearity=lasagne. nonlinearities.softmax)
l_out_target = l_out

"""
# input is the board, now 2d
X_sym = T.tensor3()
# make the new board the input
y_sym = T.tensor3()

rew = T.scalar()
distance = T.scalar()
distanceNew = T.scalar()

# output is a vector with a probability distribution
output = lasagne.layers.get_output(l_out, X_sym)
# prediction is a move with highest probability
prediction = output.argmax(-1)
predictionValue = output.max()

# output_target is a vector with a probability distribution for another input
output_target = lasagne.layers.get_output(l_out_target, y_sym)
# get best move in new state
prediction_target = output_target.argmax(-1)
predictionValue_target = output_target.max()

# the predictions functions
f_predict = theano.function([X_sym], prediction)
f_predict_target = theano.function([y_sym], prediction_target)
f_predictValue = theano.function([X_sym], predictionValue)
f_predictValue_target = theano.function([y_sym], predictionValue_target)

#rewardFunction = distance - distanceNew - 1
#rewardFunctionT = theano.function([distance, distanceNew], rewardFunction)

# q learning loss
loss = lasagne.objectives.squared_error(predictionValue_target + rew, predictionValue)

# get weights of network
params = lasagne.layers.get_all_params(l_out, trainable=True)

# update params
grad = T.grad(loss, params)
updates = lasagne.updates.sgd(grad, params, learning_rate=0.5)

# calculate loss based on used variables
f_train = theano.function([X_sym, y_sym, rew], loss, updates=updates)

weights = l_out.W.get_value()
print weights[5]

for i in range(1000):
	boardSetup(1)
	setPieceRandom('K')
	boardR = np.reshape(board, (1,64,1))
	predicted_move = f_predict(boardR)
	x, y = getCoor('K')
	moveKing(predicted_move)
	xNew, yNew = getCoor('K')
	reward = float(y - yNew)*2
	boardR2 = np.reshape(board, (1,64,1))

	loss = f_train(boardR, boardR2, reward)
	#printBoard()

weights2 = l_out.W.get_value()
print weights2[5]
"""

"""
# test
boardSetup(1)
setPieceRandom('K')

printBoard()
boardSetup(1)
setPieceRandom('K')

printBoard()
"""

"""
boardSetup(1)
setPieceRandom('K')

printBoard()
boardR = np.reshape(board, (1,64,1))
predicted_move = f_predict(boardR)
print predicted_move
moveKing(predicted_move)

printBoard()
boardR = np.reshape(board, (1,64,1))
predicted_move = f_predict(boardR)
print predicted_move
moveKing(predicted_move)

printBoard()
boardR = np.reshape(board, (1,64,1))
predicted_move = f_predict(boardR)
print predicted_move
moveKing(predicted_move)

printBoard()

"""
"""
update parameters van de ene naar de ander
"""

X_sym = T.tensor3()
y_sym = T.ivector()
rew = T.scalar()
disc = T.scalar()

# Trainable network
output = lasagne.layers.get_output(l_out, X_sym)
pred = output.argmax(-1)
max_ = output.max()
params = lasagne.layers.get_all_params(l_out)

# Target network
l_out_target = l_out
output_target = lasagne.layers.get_output(l_out_target, X_sym)
params2 = lasagne.layers.get_all_params(l_out_target)
pred_target = output.argmax(-1)
pred_target_max = output.max()

# Theano functions
loss = T.mean((max_ - (rew + (disc * pred_target_max)))**2)
grad = T.grad(loss, params)
updates = lasagne.updates.sgd(grad, params, learning_rate=0.5)

f_train = theano.function([pred_target_max, max_, X_sym, rew, disc], loss, updates=updates, allow_input_downcast=True)
f_max = theano.function([X_sym], max_, allow_input_downcast=True)
f_predict = theano.function([X_sym], pred, allow_input_downcast=True)
f_predict_target = theano.function([X_sym], pred_target, allow_input_downcast=True)
f_max_target = theano.function([X_sym], pred_target_max, allow_input_downcast=True)
grad_calc = theano.function([pred_target_max, max_, X_sym, rew, disc], grad)

weights = l_out.W.get_value()
print weights[5], weights[2]

average_loss = 0
average_reward = 0
for i in range(500):
	boardSetup(1)
	setPiece('K', 4, 4)
	state = np.reshape(board, (1,64,1))
	predicted_move = f_predict(state)
	x, y = getCoor('K')
	epsilon = 0.9
	if random.uniform(0,1) < epsilon:
		moveKingRandom()
	if random.uniform(0,1) > epsilon:
		moveKing(predicted_move)
	xNew, yNew = getCoor('K')
	if y > yNew:
		reward = float(y - yNew)
	newState = np.reshape(board, (1,64,1))

	DISCOUNT = 1
	target_max = f_max_target(newState)
	network_max = f_max(state)
	if isTerminalState():
		loss = f_train(0, network_max, state, reward, 0)
	else: 
		loss = f_train(target_max, network_max, state, reward, DISCOUNT)
	average_loss += loss
	average_reward += reward
	#print average_loss/i

print average_reward
print average_loss
gradcalcs =  grad_calc(target_max, network_max, state, reward, DISCOUNT)
#print T.norm(gradcalcs)
print gradcalcs
weights2 = l_out.W.get_value()
print weights2[5], weights2[2]

# to test
boardSetup(1)
setPiece('K', 4, 4)
printBoard()
boardR = np.reshape(board, (1,64,1))
predicted_move = f_predict(boardR)
print predicted_move
moveKing(predicted_move)
printBoard()

"""
average loss zou omlaag moeten gaan
"""
