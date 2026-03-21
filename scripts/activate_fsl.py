# reference if I need to activate fsl through ssh tunnelactivate_fsl () {

# activate_fsl () {
#         FSLDIR=~/fsl 
#         export FSLDIR
#         . ${FSLDIR}/etc/fslconf/fsl.sh
# }
# export PATH="${FSLDIR}/share/fsl/bin:${PATH}"

import os
import sys

os.environ['FSLDIR'] = "/home/srs-9/fsl"
os.environ['FSLOUTPUTTYPE'] = "NIFTI_GZ"

os.environ['FSLMULTIFILEQUIT'] = "TRUE"

os.environ['FSLTCLSH'] = "${FSLDIR}/bin/fsltclsh"

os.environ['FSLWISH'] = "${FSLDIR}/bin/fslwish"
os.environ['FSL_LOAD_NIFTI_EXTENSIONS'] = "0"
os.environ['FSL_SKIP_GLOBAL'] = "0"


os.environ["PATH"] += os.pathsep + "/home/srs-9/fsl/share/fsl/bin"

# then pass the following env to subprocess

env = os.environ.copy()