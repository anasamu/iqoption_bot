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
import getpass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def header_app():
	os.system('clear')
	print(f"=========================================================")
	print(f"> Trading Bot v1.0 (BETA)")
	print(f"> Author : Anas Amu")
	print(f"=========================================================")
	print(f"> Segala resiko di tanggung sendiri")
	print(f"=========================================================")

def classify(current,future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def get_balance(iq):
	return iq.get_balance()

def get_profit(iq):
	return iq.get_all_profit()[currency]['turbo']

def cek_win(iq, id):
	return iq.check_win(id)

def check_trade_mood(iq, currency):
    iq.start_mood_stream(currency)
    mood = iq.get_traders_mood(currency)
    iq.stop_mood_stream(currency)
    return mood

def menu_bot():
	print(f"=========================================================")
	print("# BOT MENU")
	print(f"=========================================================")
	print("> 1. BOT Trade")
	print("> 2. Signal Trade")
	print("> 3. Exit Bot")
	print(f"=========================================================")
	bot_type = int(input("> Masukkan angka pilihan: "))

	return bot_type

def login(username,password,verbose = False, iq = None, checkConnection = False):
	if verbose:
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(message)s')

	if iq == None:
		print("> Mencoba login ke akun IQ Option...")
		iq=IQ_Option(username,password)
		iq.connect()

	if iq != None:
		while True:
			if iq.check_connect() == False:
				print('> Terjadi Kesalahan saat mencoba login')
				print("> Mencoba login ulang...")
				exit()
			else:
				if not checkConnection:
					print('> Login Berhasil...')
					print("=========================================================")
				break
	change_balance(iq,"PRACTICE")
	return iq

def change_balance(iq, account):
	iq.change_balance(account)

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
	SEQ_LEN = 5
	df = df.drop(columns = ["future"]) 
	
	scaler = MinMaxScaler()
	indexes = df.index
	df_scaled = scaler.fit_transform(df)
	
	df = pd.DataFrame(df_scaled,index = indexes)
	
	sequential_data = []  
	prev_days = deque(maxlen=SEQ_LEN)

	for i in df.values: 
		prev_days.append([n for n in i[:-1]])  
		if len(prev_days) == SEQ_LEN:
			sequential_data.append([np.array(prev_days), i[-1]]) 

	random.shuffle(sequential_data) 

	buys = [] 
	sells = [] 

	for seq, target in sequential_data:  
		if target == 0: 
			sells.append([seq, target])  
		elif target == 1: 
			buys.append([seq, target]) 

	random.shuffle(buys)  
	random.shuffle(sells) 

	lower = min(len(buys), len(sells))  

	buys = buys[:lower]  
	sells = sells[:lower]  
	
	
	sequential_data = buys+sells 
	random.shuffle(sequential_data)  

	X = []
	y = []

	for seq, target in sequential_data:  
		X.append(seq)  
		y.append(target) 

	return np.array(X), y  


def scaler(iq, df):
	FUTURE_PERIOD_PREDICT = 1
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
	df['rsi'] = 100-(100/(1+rs)) 
	df = df.drop(columns = {'open','min','max','avg_gain','avg_loss','L14','H14','gain','loss'})
	dataset = df.fillna(method="ffill")
	dataset = dataset.dropna()
	dataset.sort_index(inplace = True)
	main_df = dataset

	main_df.fillna(method="ffill", inplace=True) 
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
	tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
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
	opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, decay=5e-5)
	model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	model.fit(train_x, train_y,batch_size=BATCH_SIZE,epochs=EPOCHS,)
	model.evaluate(validation_x, validation_y, verbose=0)
	prediction = pd.DataFrame(model.predict(validation_x))
	return prediction

def trading(iq,prediction, currency, currency_data, config):
	bet_percent_demo = 5
	bet_percent_real = 10
	i = 0
	bid = True
	bets = []
	MONEY = get_balance(iq) 
	trade = True
	bet_percent = config[0]
	total_trading_loss = 3
	bet_money = ((bet_percent / 100) * MONEY)
	result = prediction
	profit = 0
	total_profit = 0
	total_loss = 0
	loss_profit = 0
	take_profit = 85
	netral_profit = 0
	cut_loss = -15
	real_account_check = 0
	account_test = 'DEMO'
	trade_predict = 0.5
	trade_predict_algo = 0.5
	trade_algo_reset = 0
	buy_status = ""
	total_skip = 0
	total_skip_real = 0
	total_skip_demo = 0
	total_trade_demo = 0
	total_profit_demo = 0
	total_loss_demo = 0
	total_netral_demo = 0

	total_trade_real = 0
	total_profit_real = 0
	total_loss_real = 0
	total_netral_real = 0

	prediction_correct_call = []
	prediction_correct_put = []
	total_incorrect_predict_call = 0
	total_incorrect_predict_put = 0
	trade_prediction = ""
	percent_win = 0
	percent_lose = 0
	total_loss_algo = 0
	open_posisi = 15 # default 15 menit

	while(1):
		if(get_balance(iq) < 150):
			iq.reset_practice_balance()
		
		remaning_time = 0
		purchase_time = 0
		
		# cek total profit di akun demo
		if(real_account_check != 1 and percent_win >= config[2] and i >= config[1] and percent_lose <= config[3]):
			change_balance(iq,"REAL")
			MONEY = get_balance(iq) 
			if MONEY >= 15000:
				print(f"# Beralih ke account real")
				bet_percent = 10
				bet_money = ((bet_percent / 100) * MONEY)
				account_test = "REAL"
				profit = 0
				total_skip = total_skip_real
				total_profit = total_profit_real
				total_loss = total_loss_real
				netral_profit = total_netral_real
				loss_profit = 0
				real_account_check = 1
				i = total_trade_real
			else:
				change_balance(iq,"PRACTICE")
				print(f"# Saldo akun real tidak cukup")

		# CEK TOTAL LOSS DI AKUN REAL
		if(real_account_check == 1):
			if(bet_money < 15000):
				kekurangan = 15000 - bet_money
				bet_money = bet_money + kekurangan

			if(total_profit <= total_loss and total_loss >= config[5]):
				print(f"# Beralih ke account demo")
				change_balance(iq,"PRACTICE")
				MONEY = get_balance(iq) 
				bet_percent = 1
				total_trading_profit = 10
				bet_money = ((bet_percent / 100) * MONEY)
				account_test = "DEMO"
				profit = 0
				total_skip = 0
				total_profit = total_profit_demo
				total_loss = total_loss_demo
				netral_profit = total_netral_demo
				loss_profit = 0
				real_account_check = 0

		if(account_test == "DEMO"):
			bet_percent = bet_percent_demo
			currency_money = "$"
			i = total_trade_demo
			total_skip = total_skip_demo
			total_profit = total_profit_demo
			total_loss = total_loss_demo
			netral_profit = total_netral_demo

		else:
			currency_money = "IDR"
			i = total_trade_real
			total_profit = total_profit_real
			total_loss = total_loss_real
			netral_profit = total_netral_real
			if(real_account_check == 0):
				bet_percent = bet_percent_real

		laba = get_balance(iq) - MONEY
		percent_profit = ((laba / get_balance(iq)) * 100)

		# CEK PROFIT AKUN REAL
		if(percent_profit >= config[4]):
			if(real_account_check == 1):
				choice = input("# Batas profit untuk Trading hari ini sudah mencapai batas ingin melanjutkan? (yes) or (no) : ").lower()
				if(choice == "no"):
					exit()
				elif(choice == "yes"):
					print(f"# Proses Trading dilanjutkan")	
				else:
					sys.stdout.write("Silahkan pilih 'yes' atau 'no'")
		
		header_app()
		if(i > 0):

			t_trading = i - total_skip
			if(total_profit > 0):
				percent_win = ((total_profit / t_trading) * 100)
			if(total_loss > 0):
				percent_lose = ((total_loss / t_trading) * 100)

			print(f"# BOT Type : AUTO Trade")
			print(f"# Account Type : {account_test}")
			print(f"# Saldo awal Trade : {currency_money}{MONEY}")
			print(f"# Saldo anda saat ini : {currency_money}{get_balance(iq)}")

			print(f"# Total Trader : {i}x")
			print(f"# Total Skip Trader : {total_skip}x")
			print(f"# Total Profit : {total_profit}x")
			print(f"# Total Loss : {total_loss}x")
			print(f"# Total Netral : {netral_profit}x")
			print(f"# Laba/Rugi : {currency_money}{round(laba,2)}")
			print(f"# Persentase Profit : {round(percent_profit,2)}%")
			print(f"# Persentase Win : {round(percent_win,2)}%")
			print(f"# Persentase Lose : {round(percent_lose,2)}%")
			#print(f"# Trade Algoritma : {trade_predict}")
			print(f"=========================================================")
			print(f"# Peringatan! Fitur auto trade masih banyak kekurangan")
			print(f"# Loss Profit Kemungkinan bisa terjadi")
			print(f"=========================================================")
			
		print(f'> Open Market : {currency}')
		
		candle_check = iq.get_candles(currency, 900, 3, time.time())
		candle_check[0] = 'g' if candle_check[0]['open'] < candle_check[0]['close'] else 'r' if candle_check[0]['open'] > candle_check[0]['close'] else 'd'
		candle_check[1] = 'g' if candle_check[1]['open'] < candle_check[1]['close'] else 'r' if candle_check[1]['open'] > candle_check[1]['close'] else 'd'
		candle_check[2] = 'g' if candle_check[2]['open'] < candle_check[2]['close'] else 'r' if candle_check[2]['open'] > candle_check[2]['close'] else 'd'

		analisis = candle_check[0] + ' ' + candle_check[1] + ' ' + candle_check[2]

		if analisis.count('g') > analisis.count('r') and analisis.count('d') == 0 :
			buy_status = 'put'
		if analisis.count('r') > analisis.count('g') and analisis.count('d') == 0 :
			buy_status = 'call'

		trade_id = 0
		trade = False
		trade_correct_result = result[0][0]
		
		if(trade_correct_result in prediction_correct_put and buy_status == "put"):
			print(f'> Open Posisi : PUT')
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time < 3 or purchase_time == 30:
					done,id = iq.buy(float(bet_money),currency,"put", open_posisi)
					if not done:
						print('> Analisa gagal. tidak dapat melakukan open posisi')
					else:
						trade_prediction = 'put'
						trade_id = id
						trade = True
					break
		elif(trade_correct_result in prediction_correct_call  and buy_status == "call"):
			print(f'> Open Posisi : CALL')
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time < 3 or purchase_time == 30:
					done,id = iq.buy(float(bet_money),currency,"call", open_posisi)
					if not done:
						print('> Analisa gagal. tidak dapat melakukan open posisi')
					else:
						trade_prediction = 'call'
						trade_id = id
						trade = True
					break
		elif trade_correct_result < trade_predict_algo and buy_status == "put":
			print(f'> Open Posisi : PUT')
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time < 3 or purchase_time == 30:
					done,id = iq.buy(float(bet_money),currency,"put", open_posisi)
					if not done:
						print('> Analisa gagal. tidak dapat melakukan open posisi')
					else:
						trade_prediction = 'put'
						trade_id = id
						trade = True
					break
		elif trade_correct_result > trade_predict_algo and buy_status == "call":
			print(f'> Open Posisi : CALL')
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time < 3 or purchase_time == 30:
					done,id = iq.buy(float(bet_money),currency,"call", open_posisi)
					if not done:
						print('> Analisa gagal. tidak dapat melakukan open posisi')
					else:
						trade_prediction = 'put'
						trade_id = id
						trade = True
					break

		if trade:
			print(f'> Bet Open Posisi : {currency_money}{round(bet_money)}')	
			print(f"=========================================================")
			print("# Trading sedang berlangsung...")
			print("# Menunggu Trading berakhir...")
			print(f"=========================================================")
			print(f"# Menunggu analisa hasil trading...")
			tempo = datetime.datetime.now().second
			while(tempo != 1):
				tempo = datetime.datetime.now().second

			betsies = iq.get_optioninfo_v2(1)
			betsies = betsies['msg']['closed_options']
			for bt in betsies:
					bets.append(bt['win'])

			win = bets[-1:]
			print(f"=========================================================")
			check_profit = iq.check_win_v3(trade_id)
			if(int(check_profit) > 0):
				total_profit = total_profit + 1
				if(real_account_check == 1):
					total_profit_real = total_profit
				else:
					total_profit_demo = total_profit

				bet_percent = bet_percent + 0.05
				bet_money = (bet_percent / 100 * get_balance(iq))
				print(f"> Total profit : {total_profit}x")
				print(f"> Profit : {currency_money}{round(check_profit,2)}")
			elif(int(check_profit) < 0):
				total_loss_algo = total_loss_algo + 1
				if(trade_prediction != ""):
					if(trade_prediction == "put"):
						if(trade_correct_result not in prediction_correct_put):
							total_incorrect_predict_put = total_incorrect_predict_put + 1
							prediction_correct_put.append(trade_correct_result)
						else:
							prediction_correct_put.remove(trade_correct_result)
					elif(trade_prediction == "call"):
						if(trade_correct_result not in prediction_correct_call):
							total_incorrect_predict_call = total_incorrect_predict_call + 1
							prediction_correct_call.append(trade_correct_result)
						else:
							prediction_correct_call.remove(trade_correct_result)
				total_loss = total_loss + 1
				loss_profit = round(bet_money,2)

				if(real_account_check == 1):
					total_loss_real = total_loss
				else:
					total_loss_demo = total_loss
					
				bet_percent = bet_percent - 0.05
				bet_money = (bet_percent / 100 * get_balance(iq))
				print(f"> Total loss : {total_loss}x")
				print(f"> Loss : {currency_money}{round(check_profit,2)}")
			else:
				netral_profit = netral_profit + 1
				if(real_account_check == 1):
					total_netral_real = netral_profit
				else:
					total_netral_demo = netral_profit
				print(f"> Hasil Netral : -")
				bets.append(0)	
		else:
			total_skip = total_skip + 1
			if(real_account_check == 1):
				total_skip_real = total_skip
			else:
				total_skip_demo = total_skip
			print(f"=========================================================")
			print("# Melewati Open Posisi")


		print(f"=========================================================")	
		print("# Persiapan mengambil data candle stick...")

		while True:
			remaning_time = iq.get_remaning(ducurrencyn)
			purchase_time = remaning_time-30
			if purchase_time <= 5 or purchase_time == 35:
				print(f"=========================================================")
				print("# Sedang memuat data analisis berikutnya...")
				candle = get_all_candles(iq,currency,candle_interval,total_candle)
				result = scaler(iq,candle)
				i = i + 1
				if(real_account_check == 1):
					total_trade_real = i
				else:
					total_trade_demo = i
				break


def signal(iq, prediction, currency, currency_data):
	total_trade = 0
	prediction_correct_call = []
	prediction_correct_put = []
	total_incorrect_predict_call = 0
	total_incorrect_predict_put = 0
	total_predict_profit = 0
	total_predict_loss = 0
	total_predict_netral = 0
	trade_predict_algo = 0.5
	total_loss = 3
	trade_algo_reset = 3
	result = prediction
	while(1):

		header_app()
		if(total_trade > 0):
			print(f"# BOT Type : SIGNAL")
			print(f"# Total Analisis : {total_trade}x")
			print(f"# Total Gagal Analisis Open Posisi CALL : {total_incorrect_predict_call}x")
			print(f"# Total Gagal Analisis Open Posisi PUT : {total_incorrect_predict_put}x")
			print(f"# Total Analisis Profit : {total_predict_profit}x")
			print(f"# Total Analisis Loss : {total_predict_loss}x")
			print(f"# Total Analisis Netral : {total_predict_netral}x")
			print(f"=========================================================")
			print(f"# Peringatan! Fitur BOT SIGNAL masih banyak kekurangan")
			print(f"# Hasil Analisis Kemungkinan tidak akurat")
			print(f"# Keputusan untuk open posisi ada di tangan anda!")
			print(f"=========================================================")
		print(f'> Open Market : {currency}')
		
		candle_check = iq.get_candles(currency, 900, 3, time.time())
		candle_check[0] = 'g' if candle_check[0]['open'] < candle_check[0]['close'] else 'r' if candle_check[0]['open'] > candle_check[0]['close'] else 'd'
		candle_check[1] = 'g' if candle_check[1]['open'] < candle_check[1]['close'] else 'r' if candle_check[1]['open'] > candle_check[1]['close'] else 'd'
		candle_check[2] = 'g' if candle_check[2]['open'] < candle_check[2]['close'] else 'r' if candle_check[2]['open'] > candle_check[2]['close'] else 'd'

		analisis = candle_check[0] + ' ' + candle_check[1] + ' ' + candle_check[2]

		if analisis.count('g') > analisis.count('r') and analisis.count('d') == 0 :
			buy_status = 'put'
		if analisis.count('r') > analisis.count('g') and analisis.count('d') == 0 :
			buy_status = 'call'

		trade_correct_result = result[0][0]
		if(trade_correct_result in prediction_correct_put):
			trade_predict = 'put'
			trade = True
			print(f'> Hasil Analisis Open Posisi : PUT')
		elif(trade_correct_result in prediction_correct_call):
			trade_predict = 'call'
			trade = True
			print(f'> Hasil Analisis Open Posisi : CALL')
		elif trade_correct_result < trade_predict_algo and buy_status == "put":
			trade_predict = 'put'
			trade = True
			print(f'> Hasil Analisis Open Posisi : PUT')
		elif trade_correct_result > trade_predict_algo and buy_status == "call":
			trade_predict = 'call'
			trade = True
			print(f'> Hasil Analisis Open Posisi : CALL')
		else:
			trade = False

		if trade:
			print(f"=========================================================")
			print("# Menunggu Trading berakhir...")
			remaning_time = iq.get_remaning(ducurrencyn)
			purchase_time = remaning_time-30
			time.sleep(purchase_time)
			print(f"=========================================================")
			print(f"# Menunggu analisa hasil trading...")
			tempo = datetime.datetime.now().second
			while(tempo != 1):
				tempo = datetime.datetime.now().second
			print(f"=========================================================")
			print(f"# Silahkan masukkan hasil trading yang berlangsung.")
			print(f"> Apakah hasil analisis trading benar? (yes) or (no)")
			result_predict = input("> Hasil Analisa : ").lower()
			total_incorrect_predict_call = 0
			total_incorrect_predict_put = 0
			total_predict_profit = 0
			total_predict_loss = 0
			total_predict_netral = 0
			# jika analisa bot salah
			if(result_predict == 'no'):
				total_predict_loss = total_predict_loss + 1
				if(trade_predict == "put"):
					total_incorrect_predict_put = total_incorrect_predict_put + 1
					prediction_correct_put.append(trade_correct_result)
				elif(trade_predict == "call"):
					total_incorrect_predict_call = total_incorrect_predict_call + 1
					prediction_correct_call.append(trade_correct_result)
				else:
					total_predict_netral = total_predict_netral + 1
			else:
				total_predict_profit = total_predict_profit + 1
			
			if(total_loss >= total_predict_loss):
				if(trade_algo_reset >= 3):
					if(trade_predict_algo > 0.6):
						trade_predict_algo = 0.5
					trade_predict_algo = trade_predict_algo + 0.0001
					print("# Mengganti Algoritma trade")		
		else:
			print(f"=========================================================")
			print("# Analisa gagal. Mempersiapkan analasis berikutnya")

	print(f"=========================================================")	
	print("# Persiapan mengambil data candle stick...")

	while True:
		remaning_time = iq.get_remaning(ducurrencyn)
		purchase_time = remaning_time-30
		if purchase_time<=50:
			print(f"=========================================================")
			print("# Sedang memuat data analisis berikutnya...")
			candle = get_all_candles(iq,currency,candle_interval,total_candle)
			result = scaler(iq,candle)
			total_trade = total_trade + 1
			trade_algo_reset = trade_algo_reset + 1
			break

header_app()
currency_data = ['EURUSD','EURGBP','USDCHF','NZDUSD','USDINR','AUDCAD','USDSGD','USDHKD','GBPJPY','GBPUSD','EURJPY','AUDUSD']
ducurrencyn = 1 # menit
candle_interval = 15 # 15 detik
total_candle = 1000
config = []
username = input("> Username: ").lower()
password = getpass.getpass('> Password: ')
iq = login(username = username, password = password)

print(f"=========================================================")
print("# Silahkan tunggu...")

while True:
	header_app()
	bot_type = menu_bot()
	if(bot_type == 1 or bot_type == 2):
		currency = input('> Silahkan Pilih Market: ').upper()

		if(bot_type == 1):
			print("> Berapa Persen Biaya Open Posisi dari modal yang ingin digunakan?")
			open_posisi_prices = int(input('> Masukkan disini: '))
			print("> Minimal Berapa kali BOT untuk melakukan trader agar beralih ke akun real?")
			minimum_trader = int(input('> Masukkan disini: '))
			print("> Minimal Berapa Persen profit di akun demo untuk beralih ke akun real?")
			minimum_profit_demo = int(input('> Masukkan disini: '))
			print("> Maksimal Berapa Persen loss di akun demo untuk beralih ke akun real?")
			max_loss_demo = int(input('> Masukkan disini: '))

			print("> Minimal Berapa Persen profit di akun real untuk berhenti melakukan trade?")
			minimum_take_profit = int(input('> Masukkan disini: '))

			print("> Minimal Berapa kali loss profit di akun real untuk beralih ke akun demo?")
			minimum_cut_loss = int(input('> Masukkan disini: '))
			
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time<=5 or purchase_time == 30:
					config.append(open_posisi_prices)
					config.append(minimum_trader)
					config.append(minimum_profit_demo)
					config.append(max_loss_demo)
					config.append(minimum_take_profit)
					config.append(minimum_cut_loss)

					candle = get_all_candles(iq,currency,candle_interval,total_candle)
					prediction = scaler(iq,candle)
					trading(iq, prediction, currency, currency_data, config)
					break
		else:
			while True:
				remaning_time = iq.get_remaning(ducurrencyn)
				purchase_time = remaning_time-30
				if purchase_time<=5 or purchase_time == 30:
					candle = get_all_candles(iq,currency,candle_interval,total_candle)
					prediction = scaler(iq,candle)
					signal(iq, prediction, currency, currency_data)
					break
		
	elif(bot_type == 3):
		print("Terimakasih sudah menggunakan bot ini.")
		exit()
		break
	else:
		print("> Menu yang anda pilih tidak tersedia")

	time.sleep(0.5)
