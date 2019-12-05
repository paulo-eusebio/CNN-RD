# CNN-RD
End-to-end Image Compresion Optimization with CNNs

## About
Coding Framework complemented with the two following Convolutional Neural Networks (CNNs).
  - CNN-CR for down-sampling before image coding.
  - CNN-SR for up-sampling afer image decoding.

This framework allows to train both CNNs with a loss function that minimizes both distortion (with MSE) and rate. The former is achieved by estimating the Discrete Cosine Transform coefficients that JPEG would quantize to zero.

See *extended_abstract.pdf* for further information.

## Credits
This project was developed at [Instituto de Telecomunicações](https://it.pt) (IT) and [Instituto Superior Técnico](https://tecnico.ulisboa.pt/) in a Master Thesis context.

## Instructions 
Below are the instructions to run the provided framework.

#### Training the CNNs
Run *train.py* without any arguments Datasets, Hyper-parameters and settings are all hardcoded and defined at the beggining of the script. All these settings are commented to help change them if necessary.

#### Inferece
During trainig, the obtained models are evaluated every epoch.

For indepedent inferece (i.e. without running the training script) run *eval.py* without any arguments Datasets and settings are all hardcoded and defined at the beggining of the script. 

## Contacts
For any question or problem, please contact *paulomreusebio@hotmail.com* or open an issue.
