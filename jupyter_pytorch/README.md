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
