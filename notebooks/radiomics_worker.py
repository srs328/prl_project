
import json
import sys
from radiomics import featureextractor


def extract_radiomic_features(image_path, mask_path, params_file):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

    result = extractor.execute(
        imageFilepath=image_path,
        maskFilepath=mask_path,
    )

    return {
        key[len("original_"):]: float(value)
        for key, value in result.items()
        if key.startswith("original_")
    }


if __name__ == "__main__":
    input_json_path, output_json_path = sys.argv[1:3]

    with open(input_json_path) as f:
        args = json.load(f)

    features = extract_radiomic_features(
        image_path=args["image_path"],
        mask_path=args["mask_path"],
        params_file=args["params"],
    )

    with open(output_json_path, "w") as f:
        json.dump(features, f, indent=4)
