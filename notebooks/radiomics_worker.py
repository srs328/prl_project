import json
import sys
from radiomics import featureextractor


def extract_radiomic_features(maskName, imageName, params_file=None):
    extractor = featureextractor.RadiomicsFeatureExtractor(params_file)
    result = extractor.execute(imageName, maskName)
    features = {
        k[len("original_"):]: float(v) for k,v in result.items() if k.startswith("original_")
    }
    return features


if __name__ == "__main__":
    # Get the input/output JSON paths from command line arguments
    input_json_path = sys.argv[1]
    output_json_path = sys.argv[2]
    
    # Load input arguments
    with open(input_json_path, 'r') as f:
        args = json.load(f)
        
    # Execute the pyradiomics function
    features = extract_radiomic_features(args['image_path'], args['mask_path'], args['params'])
    
    # Save the output data back to the file system
    with open(output_json_path, 'w') as f:
        json.dump(features, f, indent=4)