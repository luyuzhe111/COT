#!/bin/bash

wget -O fmow.zip https://worksheets.codalab.org/rest/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab/contents/blob/

mv fmow.zip ../data

cd ../data

tar -xf fmow.zip

mkdir fmow_v1.1

mv images fmow_v1.1
mv country_code_mapping.csv fmow_v1.1
mv rgb_metadata.csv fmow_v1.1
mv fmow.zip fmow_v1.1
mv LICENSE fmow_v1.1
mv RELEASE_v1.1.txt fmow_v1.1
mv README.md fmow_v1.1