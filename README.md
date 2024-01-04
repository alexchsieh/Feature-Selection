# Feature Selection
Implemented the Feature Selection Method using Forward Selection and Backward Elimination.

Had heavy inspiration from my classmates & Professor Eamonn Keogh from UCR.

## Installation

clone the repo

```sh
https://github.com/alexchsieh/Feature-Selection.git
```

## Tools

install pip for pandas

```sh
sudo apt install python3-pip
```

install pandas

```sh
pip install pandas
```


## Usage example

run the program

```sh
python3 main.py
```

use a testing data set

```
./tests/Small_Data.txt
```

sample forward selection solution

```sh
Finished search! The best feature subset is [6, 1], which has an accuracy of 95.0%
```

sample backward elimination solution

```sh
Finished search! The best feature subset to remove is [5, 2, 4, 3], which has an accuracy of 95.0%
```

## Contributing

1. Fork it (<https://github.com/alexchsieh/Feature-Selection/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
