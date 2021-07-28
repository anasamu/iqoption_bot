import datetime
import time
from iqoptionapi.stable_api import IQ_Option
import pandas as pd
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

ducurrencyn = 1 # menit
ducurrencyn_sec = 60 * ducurrencyn # detik
candle_interval = 30 # menit
total_candle = 300
currency 	= 'EURUSD'
FUTURE_PERIOD_PREDICT = 1
SEQ_LEN = 5

def header_app():
	os.system('cls||clear')
	print(f"================================================")
	print(f"> Trading Bot v1.0 (BETA)")
	print(f"> Author : Anas Amu")
	print(f"================================================")
	print(f"> Segala resiko di tanggung sendiri")
	print(f"================================================")

def classify(current,future):
		if float(future) > float(current):
				return 1
		else:
				return 0

def higher(iq,Money,Actives):
		done,id = iq.buy(Money,Actives,"call", ducurrencyn)
		if not done:
				print('Error call')
				print(done, id)
				exit(0)
		
		return id

def lower(iq,Money,Actives):
		done,id = iq.buy(Money,Actives,"put", ducurrencyn)
		if not done:
				print('Error put')
				print(done, id)
				exit(0)
		
		return id

def get_balance(iq):
		return iq.get_balance()

def get_profit(iq):
		return iq.get_all_profit()[currency]['turbo']

def cek_win(iq, id):
	return iq.check_win(id)

def login(verbose = False, iq = None, checkConnection = False):

		if verbose:
				logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

		if iq == None:
			print("Mencoba login ke akun IQ Option...")
			iq=IQ_Option('anasamu7@gmail.com','smkn1gtlo.')
			iq.connect()

		if iq != None:
			while True:
				if iq.check_connect() == False:
					print('Terjadi Kesalahan saat mencoba login')
					print("Mencoba login ulang...")
					iq.connect()
				else:
					if not checkConnection:
						print('Login Berhasil...')
					break
				time.sleep(3)

		iq.change_balance("PRACTICE")
		return iq

def get_all_candles(iq,Actives,interval = 15, candle_count = 1000):
		current = iq.get_candles(Actives, interval, candle_count,time.time())
		main = pd.DataFrame()
		useful_frame = pd.DataFrame()
		for candle in current:
				useful_frame = pd.DataFrame(list(candle.values()),index = list(candle.keys())).T.drop(columns = ['at'])
				useful_frame = useful_frame.set_index(useful_frame['id']).drop(columns = ['id','from','to'])
				main = main.append(useful_frame)
				main.drop_duplicates()
		return main

def preprocess_df(df):
		df = df.drop("future", 1) 
		

		scaler = MinMaxScaler()
		indexes = df.index
		df_scaled = scaler.fit_transform(df)
		
		df = pd.DataFrame(df_scaled,index = indexes)
		
		sequential_data = []  # this is a list that will CONTAIN the sequences
		prev_days = deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in

		for i in df.values:  # iterate over the values
				prev_days.append([n for n in i[:-1]])  # store all but the target
				if len(prev_days) == SEQ_LEN:  # make sure we have 60 sequences
						sequential_data.append([np.array(prev_days), i[-1]]) 

		random.shuffle(sequential_data)  # shuffle for good measure.

		buys = []  # list that will store our buy sequences and targets
		sells = []  # list that will store our sell sequences and targets

		for seq, target in sequential_data:  # iterate over the sequential data
				if target == 0:  # if  put
						sells.append([seq, target])  # append to sells list
				elif target == 1:  # if call
						buys.append([seq, target]) 

		random.shuffle(buys)  
		random.shuffle(sells)  # shuffle 

		
		lower = min(len(buys), len(sells))  

		buys = buys[:lower]  
		sells = sells[:lower]  
		
		
		sequential_data = buys+sells  # add them together
		random.shuffle(sequential_data)  # another shuffle

		X = []
		y = []

		for seq, target in sequential_data:  
				X.append(seq)  # X is the sequences
				y.append(target)  # y is the targets

		return np.array(X), y  


def scaler(iq, df):
		
	df.isnull().sum().sum()
	df.fillna(method="ffill", inplace=True)
	df.dropna(inplace=True)
	df['future'] = df['close'].shift(-FUTURE_PERIOD_PREDICT)
	df['MA_20']   = df['close'].rolling(window = 20).mean() #moving average 20
	df['MA_50']   = df['close'].rolling(window = 50).mean() #moving average 50

	df['L14'] = df['min'].rolling(window=14).min()
	df['H14'] = df['max'].rolling(window=14).max()
	df['%K'] = 100*((df['close'] - df['L14']) / (df['H14'] - df['L14']) ) #stochastic oscilator
	df['%D'] = df['%K'].rolling(window=3).mean()

	df['EMA_20'] = df['close'].ewm(span = 20, adjust = False).mean() #exponential moving average
	df['EMA_50'] = df['close'].ewm(span = 50, adjust = False).mean()
	
	rsi_period = 14 
	chg = df['close'].diff(1)
	gain = chg.mask(chg<0,0)
	df['gain'] = gain
	loss = chg.mask(chg>0,0)
	df['loss'] = loss
	avg_gain = gain.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()
	avg_loss = loss.ewm(com = rsi_period - 1, min_periods = rsi_period).mean()

	df['avg_gain'] = avg_gain
	df['avg_loss'] = avg_loss
	rs = abs(avg_gain/avg_loss)
	df['rsi'] = 100-(100/(1+rs)) #rsi index
	df = df.drop(columns = {'open','min','max','avg_gain','avg_loss','L14','H14','gain','loss'}) #drop columns that are too correlated or are in somehow inside others
	df = df.dropna()
	dataset = df.fillna(method="ffill")
	dataset = dataset.dropna()
	dataset.sort_index(inplace = True)
	main_df = dataset

	main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
	main_df.dropna(inplace=True)

	main_df['target'] = list(map(classify, main_df['close'], main_df['future']))

	main_df.dropna(inplace=True)

	main_df['target'].value_counts()

	main_df.dropna(inplace=True)

	main_df = main_df.astype('float32')

	times = sorted(main_df.index.values)
	last_5pct = sorted(main_df.index.values)[-int(0.1*len(times))]

	validation_main_df = main_df[(main_df.index >= last_5pct)]
	main_df = main_df[(main_df.index < last_5pct)]

	train_x, train_y = preprocess_df(main_df)
	validation_x, validation_y = preprocess_df(validation_main_df)

	train_y = np.asarray(train_y)
	validation_y = np.asarray(validation_y)
	LEARNING_RATE = 0.001
	EPOCHS = 20
	BATCH_SIZE = 16 
	NAME = f"{LEARNING_RATE}-{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-{EPOCHS}-{BATCH_SIZE}-PRED-{int(time.time())}"

	earlyStoppingCallback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
	model = Sequential()
	model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(LSTM(128))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(2, activation='softmax'))


	opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=1e-6)

	# Compile model
	model.compile(
			loss='sparse_categorical_crossentropy',
			optimizer=opt,
			metrics=['accuracy']
	)

	tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


	filepath = "LSTM-best"  # unique file name that will include the epoch and the validation acc for that epoch
	checkpoint = ModelCheckpoint("models/{}.model".format(filepath), monitor='val_acc', verbose=1, save_best_only=True, mode='max') # saves only the best ones

	# Train model
	history = model.fit(
			train_x, train_y,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			validation_data=(validation_x, validation_y),
			callbacks=[tensorboard, checkpoint, earlyStoppingCallback],
	)

	score 	= model.evaluate(validation_x, validation_y, verbose=0)
	prediction = pd.DataFrame(model.predict(validation_x))
	return prediction

def trading(iq,prediction, currency):
	i = 0
	bid = True
	bets = []
	win_amm = 0
	MONEY = get_balance(iq) 
	trade = True
	martingale = 2
	bet_percent = 10
	total_trading_loss = 3
	total_trading_profit = 10
	bet_money = (bet_percent / 100 * MONEY)
	result = prediction
	profit = 0
	total_profit = 0
	total_loss = 0
	loss_profit = 0
	currency_data = ['EURUSD','EURGBP','GBPJPY','GBPUSD','EURJPY','AUDUSD']

	while(1):

		laba = get_balance(iq) - MONEY
		header_app()
		print(f"# Saldo awal Trade : {MONEY}")
		print(f"# Saldo anda saat ini : {get_balance(iq)}")
		print(f"# total profit : {total_profit}x")
		print(f"# total loss : {total_loss}x")
		print(f"# Laba/Rugi : ${laba}")
		print(f"================================================")
		if(total_loss > 0):
			bet_money = (bet_percent / 100 * MONEY)
		
		total_profit_end = total_profit - total_loss
		total_end = (MONEY + profit - loss_profit)
		
		if(total_loss > total_trading_loss):
			currency_lama = currency
			currency_data.remove(currency)
			currency = random.choice(currency_data)
			print(f"# Total lost pada pasar {currency_lama} saat ini {total_loss}x, system akan berganti keperdagangan pasar yang baru : {currency}")
			
			if(total_loss > total_profit):
				print(f"# Total loss untuk trading hari ini sudah mencapai batas..")
				exit()
			else:
				total_loss = 0

		if(total_profit_end >= total_trading_profit):
			
			print(f"# Batas profit untuk Trading hari ini sudah mencapai batas..")
			exit()

		if result[0][0] > 0.5 :
				print(f'> Open Posisi : PUT')
				trade_id = lower(iq,bet_money,currency)
				trade = True
		elif result[0][0] < 0.5 :
				print(f'> Open Posisi : CALL')
				trade_id = higher(iq,bet_money,currency) 
				trade = True
		else:
				trade_id = 0
				trade = False

		if trade:

			print(f'> Bet Open posisi : ${bet_money}')	
			print(f"================================================")
			print("# Trading sedang berlangsung...")
			
			print("# Menunggu Trading berakhir...")
			time.sleep(ducurrencyn_sec + 20)
			
			print(f"# Menunggu analisa hasil trading...")
			tempo = datetime.datetime.now().second
			while(tempo != 1):
				tempo = datetime.datetime.now().second

			betsies = iq.get_optioninfo_v2(1)
			betsies = betsies['msg']['closed_options']
			for bt in betsies:
					bets.append(bt['win'])
					win_amm = bt['win_amount']

			win = bets[-1:]
			win_amount = win_amm
			print(f"================================================")
			if(win == ['win']):
				# bet_percent = bet_percent + 1
				profit = float(win_amount)
				total_profit = float(total_profit) + float(1)
				#bet_money = (bet_percent / 100 * get_balance(iq))
				print(f"> Total profit : {total_profit}x")
				print(f"> Profit : ${win_amount}")
				#print(f"Memperbaharui harga open posisi berikutnya : {bet_money}")
			elif(win == ['loose']):
				loss_profit = float(bet_money)
				total_loss = float(total_loss) + float(1)
				bet_money = bet_money * martingale # martingale V3
				print(f"> Total loss : {total_loss}x")
				print(f"> Loss : ${loss_profit}")
				print(f"> Kompensasi loss profit : {bet_money}")
			else:
				print(f"> Hasil Netral : {win_amount}")
				bets.append(0)
				win_amm = 0

		print(f"================================================")	
		print("# Persiapan mengambil data candle stick...")
		time.sleep(30)
		print(f"================================================")
		print("# Sedang memuat data analisis berikutnya...")
		candle = get_all_candles(iq,currency,candle_interval,total_candle)
		result = scaler(iq,candle)

header_app()

iq = login()
candle 			= get_all_candles(iq,currency,candle_interval,total_candle)
prediction 	= scaler(iq,candle)
trading(iq, prediction, currency)
