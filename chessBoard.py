import numpy as np
import random
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import LeakyRectify

board = 0
board2D = 0
pieceDict = {'K' : 0, 'k' : 1, 'P' : 2}
turn = True
halfmoves = 0
reward = 0
discount = 0.9

# for board representation
noPieceNumber = -1.0/64.0
PieceNumber = 63.0/64.0

# reset all board parameters
def boardSetup(amountPieces):
	global board
	global turn
	global halfmoves
	global reward
	global noPieceNumber
	board = np.empty(64*amountPieces).reshape((amountPieces,8,8))
	for x in range(len(board)):
		for y in range(len(board[x])):
			board[x][y] = noPieceNumber
	turn = True
	halfmoves = 0
	reward = 0

# set piece, doesn't remove the old
def setPiece(piece, x, y):
	global board
	global pieceDict
	global PieceNumber
	board[pieceDict.get(piece)][y][x] = PieceNumber #(pieceDict.get(piece) + 1)

def setPieceRandom(piece):
	global board
	global pieceDict
	xRandom = random.randint(0,7)
	yRandom = random.randint(0,7)
	setPiece(piece, xRandom, yRandom)
	#board[pieceDict.get(piece)][yRandom][xRandom] = (pieceDict.get(piece) + 1)

def movePiece(piece, x, y):
	global board
	global pieceDict
	global turn
	global halfmoves
	global PieceNumber
	global noPieceNumber
	for n in range(x):
		for m in range(y):
			board[pieceDict.get(piece)][m][n] = noPieceNumber
	board[pieceDict.get(piece)][y][x] = PieceNumber
	halfmoves = halfmoves + 1
	if turn == True:
		turn = False
	else:
		turn = True

# moves whiteKing into direction
def moveKing(direction):
	global board
	global reward
	global PieceNumber
	global noPieceNumber
	x, y = getCoor('K')
	if direction == 0 and y != 0:
		board[0][y-1][x] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 1 and y != 0 and x != 0:
		board[0][y-1][x-1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 2 and y != 0 and x != 7:
		board[0][y-1][x+1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 3 and x != 0:
		board[0][y][x-1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 4 and x != 7:
		board[0][y][x+1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 5 and y != 7:
		board[0][y+1][x] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 6 and y != 7 and x != 0:
		board[0][y+1][x-1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	if direction == 7 and y != 7 and x != 7:
		board[0][y+1][x+1] = PieceNumber
		board[0][y][x] = noPieceNumber
		return True
	else: 
		return False
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
			if board[piece][y][x] > 0:
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
# if currentstate is legal
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

# like numpy reshape, returns flattend array of 2d board, with shape (1,64)
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

# linear regression (?)
l_in = lasagne.layers.InputLayer((1, 64))
custom_rectify = LeakyRectify(1)
l_out = lasagne.layers.DenseLayer(l_in, num_units=8, nonlinearity = custom_rectify)
all_param_values = lasagne.layers.get_all_param_values(l_out)

# same but for target, copy weights from trainable network
l_in_target = lasagne.layers.InputLayer((1, 64))
l_out_target = lasagne.layers.DenseLayer(l_in_target, num_units=8, nonlinearity = custom_rectify)
lasagne.layers.set_all_param_values(l_out_target, all_param_values)


state = T.matrix()
newState = T.matrix()
rew = T.scalar()
disc = T.scalar()
action = T.iscalar()

# trainable network
output = lasagne.layers.get_output(l_out, state)
params = lasagne.layers.get_all_params(l_out)
actionBest = output.argmax()
actionValue = output[0][action]
actionValues = output[0]

# target network
output_target = lasagne.layers.get_output(l_out_target, newState)
params_target = lasagne.layers.get_all_params(l_out_target)
actionBestTargetValue = output_target.max()

# general lasagne
loss = lasagne.objectives.squared_error(disc*actionBestTargetValue + rew, actionValue)
grad = T.grad(loss, params)
updates = lasagne.updates.sgd(grad, params, learning_rate=0.5)

# theano functions
f_train = theano.function([state, action, newState, rew, disc], loss, updates=updates, allow_input_downcast=True)
f_predict = theano.function([state], actionBest, allow_input_downcast=True)
f_predict_target_value = theano.function([newState], actionBestTargetValue, allow_input_downcast=True)
q_val = theano.function([state, action], actionValue)
q_vals = theano.function([state], actionValues)

# helper functions
grad_calc = theano.function([state, action, newState, rew, disc], grad)
output_calc = theano.function([state], output)

# epsilon-greedy, return the chosen move
def performAction(epsilon, state):
	randomNumber = random.uniform(0,1)
	moveSuceeded = False
	if randomNumber < epsilon:
		randMove = 0
		while moveSuceeded == False:
			randMove = random.randint(0,7)
			moveSuceeded = moveKing(rand)
		return randMove
	else:
		qVals = q_vals(state)
		sortedMoves = np.argsort(-qVals, axis=0)
		counter = 0
		predictedMove = 0
		# choses next best move after illegal move
		while moveSuceeded == False:
			predictedMove = sortedMoves[counter]
			moveSuceeded = moveKing(predictedMove)
		return predictedMove

# the main training loop
def trainLoop(iterations):
	average_loss = 0
	average_reward = 0
	reward = 0
	discount = 1
	counter = 0
	for i in range(iterations):
		boardSetup(1)
		#setPieceRandom('K')
		setPiece('K', 0, 0)
		state = flattenChessboard(board)

		x, y = getCoor('K')
		# epsilon decreases over time 1-(i*0.0001)
		move = performAction(0, state)
		xNew, yNew = getCoor('K')
		if y > yNew:
			reward = float(y - yNew)
		if y <= yNew:
			reward = 0

		newState = flattenChessboard(board)
		#predictedMoveTarget = f_predict_target_value(newState)
		#if isTerminalState():
			#loss = f_train(0, actionValue, state, reward, 0)
		#else:
		loss = f_train(state, move, newState, reward, discount)

		counter += i
		if counter > 10:
			all_param_values = lasagne.layers.get_all_param_values(l_out)
			lasagne.layers.set_all_param_values(l_out_target, all_param_values)
			counter = 0

		average_loss += loss
		average_reward += reward

		#q values and gradient test
		# for i in range(8):
		# 	gradcalcs = grad_calc(predictedMoveTarget, i, state, reward, discount)
		# 	a,b = gradcalcs

		# 	print '-'*80
		# 	print np.sum(a**2)
		# 	print np.sum(b**2)
		# 	print q_val(state, i)
		#print average_reward
		print average_loss/i	
	#gradcalcs = grad_calc(predictedMoveTarget, move, state, reward, discount)
	#print gradcalcs	

weights = l_out.W.get_value()
# weights2 = l_out_target.W.get_value()
#print weights[0]
# print weights2[0]

trainLoop(10)

weights2 = l_out.W.get_value()
# weights2 = l_out_target.W.get_value()
#print weights2[0]
#print weights2
# print weights2[0]

boardSetup(1)
setPiece('K', 4, 4)
state = flattenChessboard(board)

# print weights
# print state
# print output_calc(state)
# print np.dot(state, weights)

#weights2 = l_out.W.get_value()
#print weights
#print weights2
#print weights-weights2

# to test
def test(iterations, j):
	for j in range(j):
		boardSetup(1)
		#setPiece('K', 4, 4)
		setPieceRandom('K')
		printBoard()
		for i in range(iterations):
			state = flattenChessboard(board)
			#boardR = np.reshape(board, (1,64))
			predictedMove = f_predict(state)
			moveKing(predictedMove)
			printBoard()
			print '-'*50
		print '-'*100

#test(5,5)

"""
average loss zou omlaag moeten gaan
"""
