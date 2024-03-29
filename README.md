# Model-Performance-Predictor

Мы применяем алгоритмы машинного обучения в условиях постоянно меняющегося мира.
Поэтому нам необходимо понимать точность модели не только во время обучения,
но и во время её эксплуатации в промышленной среде. Условно будем называть это мониторингом модели.

Одной из проблем такого мониторинга является существенный временной
разрыв между датой прогноза и наблюдением фактического значения целевой переменной.
Иначе говоря решение мы принимаем уже в текущий момент времени и только через
достаточно большое количество времени мы поймём насколько оно было верным.

Существуют классические подходы к снижению модельных рисков, которые базируются на гипотезе,
что существенные отклонения в распределении входных факторов модели могут
сигнаризировать о непрезентативности разработанной модели для текущего потока данных.
Главным недостатком таких подходов является то, что мы смотрим на распределение каждого фактора
по отдельности, а не на совместное их распределение, на котором фактически базируется модель.

Поэтому предлагается ввести дополнительную модель, именуемую далее MPP,
чтобы она следила за основной моделью. Преимуществом такой модели можно сразу выделить то,
что для её обучения будут использоваться те же самые экзогенные данные,
что и для основной модели.

Наиболее подробно эта концепция описана в статье [MPP: Model Performance Predictor](<https://arxiv.org/abs/1902.08638>)
