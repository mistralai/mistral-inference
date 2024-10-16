# base path
ROOT_DIR_WITH_SLASH := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
MISTRAL_MODELS_PATH = $(ROOT_DIR_WITH_SLASH)mistral_models

# download from:
CODESTRAL_MAMBA_URL = https://models.mistralcdn.com/codestral-mamba-7b-v0-1/codestral-mamba-7B-v0.1.tar
MISTRAL_7B_3_URL = https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar
MATHSTRAL_7B_URL = https://models.mistralcdn.com/mathstral-7b-v0-1/mathstral-7B-v0.1.tar

# download to:
CODESTRAL_MAMBA_TAR_PATH = $(MISTRAL_MODELS_PATH)/codestral-mamba-7B-v0.1.tar
MISTRAL_7B_3_TAR_PATH = $(MISTRAL_MODELS_PATH)/mistral-7B-Instruct-v0.3.tar
MATHSTRAL_7B_TAR_PATH = $(MISTRAL_MODELS_PATH)/mathstral-7B-v0.1.tar

# extract to:
CODESTRAL_MAMBA_DIR_PATH = $(MISTRAL_MODELS_PATH)/codestral-mamba-7B-v0.1
MISTRAL_7B_3_DIR_PATH = $(MISTRAL_MODELS_PATH)/mistral-7B-Instruct-v0.3
MATHSTRAL_7B_DIR_PATH = $(MISTRAL_MODELS_PATH)/mathstral-7B-v0.1


.PHONY: mistral-models download-mistral extract-mistral

mistral-models: download-mistral extract-mistral

# Download the .tar files
download-mistral: \
	$(MISTRAL_MODELS_PATH)/codestral-mamba-7B-v0.1.tar \
	$(MISTRAL_MODELS_PATH)/mistral-7B-Instruct-v0.3.tar \
	$(MISTRAL_MODELS_PATH)/mathstral-7B-v0.1.tar

$(CODESTRAL_MAMBA_TAR_PATH):
	@echo "Downloading Codestral Mamba model..."
	mkdir -p $(MISTRAL_MODELS_PATH)
	wget -P $(MISTRAL_MODELS_PATH) $(CODESTRAL_MAMBA_URL)

$(MISTRAL_7B_3_TAR_PATH):
	@echo "Downloading Mistral 7B model..."
	mkdir -p $(MISTRAL_MODELS_PATH)
	wget -P $(MISTRAL_MODELS_PATH) $(MISTRAL_7B_3_URL)

$(MATHSTRAL_7B_TAR_PATH):
	@echo "Downloading Mathstral 7B model..."
	mkdir -p $(MISTRAL_MODELS_PATH)
	wget -P $(MISTRAL_MODELS_PATH) $(MATHSTRAL_7B_URL)

# Extract the .tar files to the target dir_paths
extract-mistral: \
	$(CODESTRAL_MAMBA_DIR_PATH) \
	$(MISTRAL_7B_3_DIR_PATH) \
	$(MATHSTRAL_7B_DIR_PATH)

$(CODESTRAL_MAMBA_DIR_PATH): $(CODESTRAL_MAMBA_TAR_PATH)
	@echo "Extracting Codestral Mamba model..."
	mkdir -p $(CODESTRAL_MAMBA_DIR_PATH)
	tar -xf $(CODESTRAL_MAMBA_TAR_PATH) -C $(CODESTRAL_MAMBA_DIR_PATH)

$(MISTRAL_7B_3_DIR_PATH): $(MISTRAL_7B_3_TAR_PATH)
	@echo "Extracting Mistral 7B model..."
	mkdir -p $(MISTRAL_7B_3_DIR_PATH)
	tar -xf $(MISTRAL_7B_3_TAR_PATH) -C $(MISTRAL_7B_3_DIR_PATH)

$(MATHSTRAL_7B_DIR_PATH): $(MATHSTRAL_7B_TAR_PATH)
	@echo "Extracting Mistral 7B model..."
	mkdir -p $(MATHSTRAL_7B_DIR_PATH)
	tar -xf $(MATHSTRAL_7B_TAR_PATH) -C $(MATHSTRAL_7B_DIR_PATH)

clean-mistral:
	rm -rf $(MISTRAL_MODELS_PATH)/*

