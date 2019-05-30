
download_data_td-frauddetection-001:
	@echo "Downloading data for project: td-frauddetection-001"
	kaggle competitions download -c talkingdata-adtracking-fraud-detection

	# Extracting
	cd ~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/\
			&& unzip train.csv.zip -d . \
			&& mv mnt/ssd/kaggle-talkingdata2/competition_files/train.csv . \
			&& unzip test.csv.zip

download_data_av-demandprediction-001: extract_data_av-demandprediction
	@echo "Downloading data for project: av-demandprediction-001"
	kaggle competitions download \
		-c avito-demand-prediction \
		-f train_jpg_0.zip train.csv.zip \
			periods_train.csv.zip train_active.csv.zip \
			test.csv.zip

extract_data_av-demandprediction:
	# Extracting
	@cd ~/.kaggle/competitions/avito-demand-prediction/\
			&& unzip train.csv.zip -d . \
			&& unzip train_active.csv.zip -d . \
			&& unzip periods_train.csv.zip -d . \
			&& unzip test.csv.zip -d .

	@echo "[warn] not uncompressing: train_jpg_0.zip"

all:
	@kaggle competitions list -s "avito"