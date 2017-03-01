#!/usr/bin/env bash

# Import and recode all annotation files. Usage:
#
#    ./import_cropobjects.sh
#
# (All paths are fixed for my system, not portable to any other machine!)

ANNOT_ROOT=/Users/hajicj/mhr/MFF-MUSCIMA-annotations/annotations
DATASET_ROOT=/Users/hajicj/data/MUSCIMA++/muscimapp-dataset-repo

DATA_ROOT=${DATASET_ROOT}/data
SCRIPTS_ROOT=${DATASET_ROOT}/tools/muscima/scripts

IMPORT_DIR=${DATA_ROOT}/cropobjects-import
RECODE_TO_DIR=${DATA_ROOT}/cropobjects

##############################################################################
if [ ! -d ${DATA_ROOT} ]; then
    mkdir -p ${DATA_ROOT}
fi
if [ ! -d ${IMPORT_DIR} ]; then
    mkdir -p ${IMPORT_DIR}
fi

echo "INFO: Copying annot files."
cp ${ANNOT_ROOT}/*/201{6,7}-*_MUSCIMA/annotations/*.xml ${DATA_ROOT}/cropobjects-import

echo "INFO: Recoding annot files."
${SCRIPTS_ROOT}/recode_cropobjects.py -r ${IMPORT_DIR} -i ${IMPORT_DIR}/* \
                                        -o ${RECODE_TO_DIR} --recode_uids -v

##############################################################################
rm -r ${IMPORT_DIR}