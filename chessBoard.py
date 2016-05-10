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
	rand = random.randint(0,7)
	moveKing(rand)
	return rand

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

# like numpy reshape
def flattenChessboard(matrix):
	newArray = np.zeros((1,64))
	array = []
	for extra in range(len(matrix)):
		for row in range(len(matrix[extra])):
			for number in range(len(matrix[extra][row])):
				newArray[0][row*8+number] = matrix[extra][row][number]
				#array.append(matrix[row][number])
	return newArray

boardSetup(1)
setPieceRandom('K')

"""
notes:

"""

"""
NETWORK
"""

# A very simple network, a single layer with one neuron per target class.
# Using the softmax activation function gives us a probability distribution at the output.
l_in = lasagne.layers.InputLayer((1, 64))
l_out = lasagne.layers.DenseLayer(l_in, num_units=8, nonlinearity=lasagne. nonlinearities.softmax)
l_out_target = l_out

X_sym = T.matrix()
y_sym = T.ivector()
rew = T.scalar()
disc = T.scalar()
move = T.iscalar()

# trainable network
output = lasagne.layers.get_output(l_out, X_sym)
output2 = lasagne.layers.get_output(l_out, X_sym, deteministic = True)
actionBest = output.argmax(-1)
actionBestValue = output.max()
params = lasagne.layers.get_all_params(l_out)
actionValue = output2[0][move]

# target network
l_out_target = l_out
output_target = lasagne.layers.get_output(l_out_target, X_sym)
params2 = lasagne.layers.get_all_params(l_out_target)
actionBestTarget = output.argmax(-1)
actionBestTargetValue = output_target.max()

# general lasagne
loss = lasagne.objectives.squared_error(disc*actionBestTargetValue + rew, actionBestValue)
grad = T.grad(loss, params)
updates = lasagne.updates.sgd(grad, params, learning_rate=0.1)

# theano functions
f_train = theano.function([actionBestTargetValue, actionBestValue, X_sym, rew, disc], loss, updates=updates, allow_input_downcast=True)
q_val_max = theano.function([X_sym], actionBestValue, allow_input_downcast=True)
q_val = theano.function([X_sym, move], actionValue)
f_predict = theano.function([X_sym], actionBest, allow_input_downcast=True)
f_predict_target = theano.function([X_sym], actionBestTarget, allow_input_downcast=True)
f_predict_target_value = theano.function([X_sym], actionBestTargetValue, allow_input_downcast=True)
grad_calc = theano.function([actionBestTargetValue, actionBestValue, X_sym, rew, disc], grad)


weights = l_out.W.get_value()
print weights[5], weights[2]

# epsilon-greedy
def performAction(epsilon, predictedMove, state):
		actionValue = 0
		if random.uniform(0,1) < epsilon:
			randMove = moveKingRandom()
			actionValue = q_val(state, randMove)
		if random.uniform(0,1) > epsilon:
			moveKing(predictedMove)
			actionValue = q_val_max(state)
		return actionValue

# the main training loop
def trainLoop(iterations):
	average_loss = 0
	average_reward = 0
	reward = 0
	discount = 1
	for i in range(iterations):
		boardSetup(1)
		setPiece('K', 4, 4)
		state = flattenChessboard(board)

		predictedMove = f_predict(state)
		x, y = getCoor('K')
		actionValue = performAction(0.9, predictedMove, state)
		xNew, yNew = getCoor('K')
		if y > yNew:
			reward = float(y - yNew)
		if y <= yNew:
			reward = 0
		newState = flattenChessboard(board)

		predictedMoveTarget = f_predict_target_value(newState)
		if isTerminalState():
			loss = f_train(0, actionValue, state, reward, 0)
		else:
			loss = f_train(predictedMoveTarget, actionValue, state, reward, discount)
		average_loss += loss
		average_reward += reward
	print average_reward
	print average_loss	
	#gradcalcs = grad_calc(target_max, network_q_value, state, reward, DISCOUNT)
	#print gradcalcs	

trainLoop(5)

print weights[5], weights[2]

# to test
boardSetup(1)
setPiece('K', 4, 4)
printBoard()
boardR = np.reshape(board, (1,64))
predicted_move = f_predict(boardR)
print predicted_move
moveKing(predicted_move)
printBoard()

"""
average loss zou omlaag moeten gaan
"""
