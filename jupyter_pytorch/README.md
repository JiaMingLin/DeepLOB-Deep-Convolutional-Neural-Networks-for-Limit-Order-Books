* run by

  ```BREVITAS_JIT=1 python quantize_train.py --settings <YAML settings file name in ./exp_cases>```

* the setting example

  ```
  exp_name: fxp_w4_acc16_a8_i16_o8_r8   # used to save model name
    
  learning_rate: 0.001
  epochs: 20
  
  #quantization
  quant: True
  quant_type: fxp     # int for general quantization, fxp for fixed point quantization(power of two scaling factor)
  no_brevitas: False
  w_bit: 4      # bitwidth of weight
  acc_bit: 16   # bitwidth of accumulation
  a_bit: 8      # bitwidth of activation
  i_bit: 16     # bitwidth of input
  o_bit: 8      # bitwidth of output
  r_bit: 8      # bitwidth of recurrent memory
  ```
* Result(Test Performance)
  ```
  Epoch 20/20, Train Loss: 0.7507,           Validation Loss: 0.9749, Duration: 1:30:26.456265, Best Val Epoch: 14
  accuracy_score: 0.5995712892865336
                precision    recall  f1-score   support
             0     0.5688    0.6125    0.5898     47915
             1     0.6211    0.6871    0.6524     48050
             2     0.6123    0.4887    0.5436     43523

      accuracy                         0.5996    139488
     macro avg     0.6007    0.5961    0.5953    139488
  weighted avg     0.6004    0.5996    0.5970    139488
  ```
