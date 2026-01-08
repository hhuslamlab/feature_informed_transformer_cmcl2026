# write a shell script to replace all "nan" in all json files with "" (empty string)
# Usage: ./replace_nan.sh
# the json files are in this location: ../../data/fixed_run/analysis/lemma_test_pred_sf

# loop through all json files in the directory
for file in ../../data/fixed_run/analysis/lemma_test_pred_sf/*.json
do
	# replace all "nan" with "" and save the changes in the same file
	sed -i "s/nan/'nan'/g" $file
	sed -i "s/(,/('nothing',/g" $file
	sed -i "s/, )/,'nothing')/g" $file
done
