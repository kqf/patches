basepath = .datasets/ground/VisDrone

$(basepath)/VisDrone2019-DET-train/annotations.json: $(basepath)/VisDrone2019-DET-train
	pytest -s tests/test_parses_visdrone.py --use-real
	cp $(basepath)/VisDrone2019-DET-train/VisDrone2019-DET-train/annotations.json $@

$(basepath)/VisDrone2019-DET-train: $(basepath)
	unzip $(basepath)/visdrone-dataset.zip -d $(basepath)

$(basepath):
	@kaggle datasets download kushagrapandya/visdrone-dataset -p $@
