# Modified QED objective
#
# Modified from original: https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/QED.py
# 
#  Copyright (c) 2009-2017, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import rdkit.Chem.QED as qd
import math

shift = -1.5 # The shift of the peak in logP score transformation
logp_modified_ads_params = qd.ADSparameter(A=3.172690585, B=137.8624751, C=2.534937431+shift, D=4.581497897, E=0.822739154, F=0.576295591, DMAX=131.3186604)

adsParameters_modified = {
  'MW': qd.ADSparameter(A=2.817065973, B=392.5754953, C=290.7489764, D=2.419764353, E=49.22325677,
                     F=65.37051707, DMAX=104.9805561),
  'ALOGP': logp_modified_ads_params,
  'HBA': qd.ADSparameter(A=2.948620388, B=160.4605972, C=3.615294657, D=4.435986202, E=0.290141953,
                      F=1.300669958, DMAX=148.7763046),
  'HBD': qd.ADSparameter(A=1.618662227, B=1010.051101, C=0.985094388, D=0.000000001, E=0.713820843,
                      F=0.920922555, DMAX=258.1632616),
  'PSA': qd.ADSparameter(A=1.876861559, B=125.2232657, C=62.90773554, D=87.83366614, E=12.01999824,
                      F=28.51324732, DMAX=104.5686167),
  'ROTB': qd.ADSparameter(A=0.010000000, B=272.4121427, C=2.558379970, D=1.565547684, E=1.271567166,
                       F=2.758063707, DMAX=105.4420403),
  'AROM': qd.ADSparameter(A=3.217788970, B=957.7374108, C=2.274627939, D=0.000000001, E=1.317690384,
                       F=0.375760881, DMAX=312.3372610),
  'ALERTS': qd.ADSparameter(A=0.010000000, B=1199.094025, C=-0.09002883, D=0.000000001, E=0.185904477,
                         F=0.875193782, DMAX=417.7253140),
}
adsParameters_names = {
  "Molecular weight": 'MW',
  "SlogP": 'ALOGP',
  "HB-donors (Lipinski)":'HBD',
  "HB-acceptors (Lipinski)": 'HBA',
  "PSA": 'PSA',
  "Number of rotatable bonds": 'ROTB',
  "Number of aromatic rings": 'AROM',
  'ALERTS': 'ALERTS'
}

# Compute modified QED
WEIGHT_MAX = qd.QEDproperties(0.50, 0.25, 0.00, 0.50, 0.00, 0.50, 0.25, 1.00)
WEIGHT_MEAN = qd.QEDproperties(0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.95)
WEIGHT_NONE = qd.QEDproperties(1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00)
WEIGHT_MEAN_NOALERTS = qd.QEDproperties(0.66, 0.46, 0.05, 0.61, 0.06, 0.65, 0.48, 0.0)
def expert_qed(mol, w=WEIGHT_MEAN_NOALERTS, qedProperties=None):
    if qedProperties is None:
        qedProperties = qd.properties(mol)
    d = [qd.ads(pi, adsParameters_modified[name]) for name, pi in qedProperties._asdict().items()]
    t = sum(wi * math.log(di) for wi, di in zip(w, d))
    return math.exp(t / sum(w))

def get_adsParameters():
    return adsParameters_names


def qed_properties_from_physchem(scoring_component_names, physchem_properties):
    raw_properties = {}
    for i, sc in enumerate(scoring_component_names):
      raw_properties[adsParameters_names[sc]] = physchem_properties[i]
      
    qedProperties = qd.QEDproperties(raw_properties['MW'], raw_properties['ALOGP'], raw_properties['HBA'], raw_properties['HBD'], raw_properties['PSA'], raw_properties['ROTB'], raw_properties['AROM'], 0)
    return qedProperties

  
