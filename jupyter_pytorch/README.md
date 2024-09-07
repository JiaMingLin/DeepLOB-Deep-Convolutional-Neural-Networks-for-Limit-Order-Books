# Run

  ```BREVITAS_JIT=1 python quantize_train.py --settings <YAML settings file name in ./exp_cases>```

# Results
## fxp_w4_acc16_a8_i16_o8_r8

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

## fxp_w8_acc8_a8_i8_o8_r8

  ```
  Epoch 20/20, Train Loss: 0.6563,           Validation Loss: 0.8900, Duration: 0:59:16.238221, Best Val Epoch: 18
  accuracy_score: 0.6302549323239275
                precision    recall  f1-score   support

             0     0.6248    0.6191    0.6220     47915
             1     0.7392    0.5868    0.6543     48050
             2     0.5579    0.6905    0.6171     43523

      accuracy                         0.6303    139488
     macro avg     0.6406    0.6321    0.6311    139488
  weighted avg     0.6433    0.6303    0.6316    139488
  ```
## fxp_w4_acc8_a8_i8_o8_r8(Zscore)

  ```
  Epoch 15/20, Train Loss: 0.5926,           Validation Loss: 0.8945, Duration: 0:16:22.321481, Best Val Epoch: 14
  accuracy_score: 0.6382556205551732
                precision    recall  f1-score   support

             0     0.6473    0.6273    0.6371     47915
             1     0.7332    0.5821    0.6489     48050
             2     0.5646    0.7124    0.6300     43523

      accuracy                         0.6383    139488
     macro avg     0.6484    0.6406    0.6387    139488
  weighted avg     0.6511    0.6383    0.6390    139488
  ```
