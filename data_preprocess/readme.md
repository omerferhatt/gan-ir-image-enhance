# Data Pre-Process
### Purpose
Raw images going to be converted to low and high contrast images for data augmentation

Example output of data pre-process:

![image](https://raw.githubusercontent.com/omerferhatt/gan-ir-image-enhance/master/images/collage.jpg)

### How to Use

```bash
$ python data_preprocess -r [raw_dir] -o [out_dir] -a [contrast] -b [brightness]
```

Example usage:

```bash
$ python data_preprocess -r raw_dir -o output_dir -a 0.1 -b 80
```