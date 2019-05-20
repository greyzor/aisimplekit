
download_data_td-frauddetection-001:
	@echo "Downloading data for project: td-frauddetection-001"
	kaggle competitions download -c talkingdata-adtracking-fraud-detection

	# Extracting
	cd ~/.kaggle/competitions/talkingdata-adtracking-fraud-detection/\
			&& unzip train.csv.zip -d . \
			&& mv mnt/ssd/kaggle-talkingdata2/competition_files/train.csv . \
			&& unzip test.csv.zip

all:
