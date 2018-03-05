DATA_DIR=${CTA_DATA}/Prod3b/Paranal
OUT_DIR=${CTA_SOFT}/tino_cta/data/prod3b/paranal_LND_edge

MODE ?= wave


features: features_gamma features_proton

features_all:
	MODE=wave make features
	MODE=tail make features


features_gamma: ${OUT_DIR}/features_${MODE}_gamma.h5
features_proton: ${OUT_DIR}/features_${MODE}_proton.h5
${OUT_DIR}/features_${MODE}_gamma.h5:
	CHANNEL=gamma make features_channel
${OUT_DIR}/features_${MODE}_proton.h5:
	CHANNEL=proton make features_channel
features_channel:
	write_feature_table.py \
	-i ${DATA_DIR} \
	-o ${OUT_DIR}/features_${MODE}_${CHANNEL}.h5 \
	--${MODE} --${CHANNEL} ${DRY}



clean: clean_features

clean_features:
	rm ${OUT_DIR}/features*.h5
