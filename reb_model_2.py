import numpy as np
import rand_destribution as rd
from collections import namedtuple
from enum import Enum
import math

Distribution = namedtuple('Distribution', ['type', 'params'])


class PrdStatus(Enum):
    TRANSMITTING = 1
    EMS_CONTROL = 2
    NOISE_PROTECTION = 3
    STOPPED = 4


class JamStatus(Enum):
    LOOKING_FOR = 1
    SUPPRESSING = 2
    STOPPED = 3


class SetModelException(Exception):

    def __str__(self, text):
        return text


def init_distrib(type, params):
    """
    Задает тип и параметры распределения интервала поступления заявок.
    Вид распределения                   Тип[types]     Параметры [params]
    Экспоненциальное                      'М'             [mu]
    Гиперэкспоненциальное 2-го порядка    'Н'         [y1, mu1, mu2]
    Гамма-распределение                   'Gamma'       [mu, alpha]
    Эрланга                               'E'           [r, mu]
    Кокса 2-го порядка                    'C'         [y1, mu1, mu2]
    Парето                                'Pa'         [alpha, K]
    Детерминированное                      'D'         [b]
    Равномерное                         'Uniform'     [mean, half_interval]
    """

    if type == "M":
        dist = rd.Exp_dist(params)
    elif type == "H":
        dist = rd.H2_dist(params)
    elif type == "E":
        dist = rd.Erlang_dist(params)
    elif type == "C":
        dist = rd.Cox_dist(params)
    elif type == "Pa":
        dist = rd.Pareto_dist(params)
    elif type == "Normal":
        dist = rd.Normal_dist(params)
    elif type == "Gamma":
        dist = rd.Gamma(params)
    elif type == "Uniform":
        dist = rd.Uniform_dist(params)
    elif type == "D":
        dist = rd.Det_dist(params)
    else:
        raise SetModelException("Неправильно задан тип распределения источника. Варианты М, Н, Е, С, Pa, Uniform")

    return dist


class Prd:
    def __init__(self, prd_distrib, ems_distrib, noise_protection_distrib, prob_right_noise_protection=0.3):

        self.prd_distrib = init_distrib(prd_distrib.type, prd_distrib.params)
        self.ems_distrib = init_distrib(ems_distrib.type, ems_distrib.params)
        self.noise_protection_distrib = init_distrib(noise_protection_distrib.type,
                                                     noise_protection_distrib.params)
        self.prob_right_noise_protection = prob_right_noise_protection

        self.status = PrdStatus.STOPPED
        self.time_to_end = 1e10
        self.success_num = 0
        self.total_prd_num = 0
        self.prd_moments = [0, 0, 0]
        self.prd_times = []
        self.start_time = 0
        self.end_time = 0

    def refresh_times_stat(self, new_a):
        for i in range(3):
            self.prd_moments[i] = self.prd_moments[i] * (1.0 - (1.0 / self.success_num)) + math.pow(new_a,
                                                                                                    i + 1) / self.success_num

    def on_start_prd(self, ttek, is_all_ok=True):
        self.status = PrdStatus.TRANSMITTING
        self.time_to_end = ttek + self.prd_distrib.generate()
        self.total_prd_num += 1
        if is_all_ok:
            self.start_time = ttek

    def on_end_prd(self, ttek):
        self.success_num += 1
        self.refresh_times_stat(ttek - self.start_time)
        self.prd_times.append(ttek - self.start_time)
        self.on_start_prd(ttek, True)

    def on_suppressed(self, ttek):
        self.status = PrdStatus.EMS_CONTROL
        self.time_to_end = ttek + self.ems_distrib.generate()

    def on_noise_protection_start(self, ttek):
        self.status = PrdStatus.NOISE_PROTECTION
        self.time_to_end = ttek + self.noise_protection_distrib.generate()

    def on_noise_protection_end(self, ttek):
        r = np.random.random()
        if r < self.prob_right_noise_protection:
            self.on_start_prd(ttek, False)
        else:
            self.on_suppressed(ttek)

    def __str__(self):
        text = "\nPRD\n"
        text += f"Status = {self.status}\n"
        text += f"Prd distribution: {self.prd_distrib.type} with params:[ {self.prd_distrib.params} ]\n"
        text += f"Ems distribution: {self.ems_distrib.type} with params: [ {self.ems_distrib.params} ]\n"
        text += f"Noise protection distribution: {self.noise_protection_distrib.type} with params: [ {self.noise_protection_distrib.params} ]\n"
        return text


class Jam:
    def __init__(self, intelligence_distrib, suppression_distrib, prob_suppression=0.2):
        self.intelligence_distrib = init_distrib(intelligence_distrib.type, intelligence_distrib.params)
        self.suppression_distrib = init_distrib(suppression_distrib.type, suppression_distrib.params)
        self.prob_suppression = prob_suppression
        self.status = JamStatus.STOPPED
        self.time_to_end = 1e10
        self.success_num = 0

    def on_start_intelligence(self, ttek):
        self.status = JamStatus.LOOKING_FOR
        self.time_to_end = ttek + self.intelligence_distrib.generate()

    def on_end_intellegence(self, ttek, prd_status):
        if prd_status == PrdStatus.TRANSMITTING:
            self.on_start_suppression(ttek)
        else:
            self.on_start_intelligence(ttek)

    def on_start_suppression(self, ttek):
        self.status = JamStatus.SUPPRESSING
        self.time_to_end = ttek + self.suppression_distrib.generate()

    def on_end_suppression(self, ttek):
        r = np.random.random()
        if r < self.prob_suppression:
            self.success_num += 1
            self.on_start_intelligence(ttek)
        else:
            self.on_start_suppression(ttek)

    def __str__(self):
        text = "\nJAM\n"
        text += f"Status = {self.status}\n"
        text += f"Intelligence distribution: {self.intelligence_distrib.type} with params:[ {self.intelligence_distrib.params} ]\n"
        text += f"Suppression distribution: {self.suppression_distrib.type} with params: [ {self.suppression_distrib.params} ]\n"
        return text


class Model:

    def __init__(self, prd_distrib, ems_distrib, noise_protection_distrib,
                 intelligence_distrib, suppression_distrib,
                 prob_suppression=0.2, prob_right_noise_protection=0.3, model_num=1000):
        self.ttek = 0
        self.prd = Prd(prd_distrib, ems_distrib, noise_protection_distrib,
                       prob_right_noise_protection=prob_right_noise_protection)
        self.jam = Jam(intelligence_distrib, suppression_distrib, prob_suppression=prob_suppression)

        self.model_num = model_num

    def run(self):
        self.prd.on_start_prd(self.ttek)
        self.jam.on_start_intelligence(self.ttek)
        iter_num = 0
        while self.prd.success_num < self.model_num:
            if self.prd.time_to_end < self.jam.time_to_end:
                earlier_time = self.prd.time_to_end
                self.ttek = earlier_time
                if self.prd.status == PrdStatus.TRANSMITTING:
                    self.prd.on_end_prd(earlier_time)
                elif self.prd.status == PrdStatus.EMS_CONTROL:
                    self.prd.on_noise_protection_start(earlier_time)
                elif self.prd.status == PrdStatus.NOISE_PROTECTION:
                    self.prd.on_noise_protection_end(earlier_time)

            else:
                earlier_time = self.jam.time_to_end
                self.ttek = earlier_time
                if self.jam.status == JamStatus.LOOKING_FOR:
                    self.jam.on_end_intellegence(earlier_time, self.prd.status)
                elif self.jam.status == JamStatus.SUPPRESSING:
                    jam_success_num = self.jam.success_num
                    self.jam.on_end_suppression(earlier_time)
                    if self.jam.success_num > jam_success_num:
                        self.prd.on_suppressed(earlier_time)

            self.print_status(iter_num)
            iter_num += 1

    def print_status(self, iter_num=0):
        print(
            f"ITER {iter_num} TTEK {self.ttek} PRD SUCCESS NUM: {self.prd.success_num}, SUPPRESSIONS NUM: {self.jam.success_num}\n\tPRD MOMENTS: {self.prd.prd_moments}")


def create_normal_bins(normal_params, size=10000):
    distrib = rd.Normal_dist(normal_params)
    bins = []
    for i in range(size):
        bins.append(distrib.generate())

    return bins


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Распределение времени передачи сигнала
    # Парметры усеченного нормального распределения указаны в следющем порядке
    #   mean - среднее значение
    #   sko - СКО
    #   min - минимальное значение(опционально)
    #   max - максимальное значение(опцианольно, должно быть задано минимальное     тоже)

    prd_distrib = Distribution('Normal', [7, 2.5, 3, 30])

    # распределение времени проверки на ЭМС
    ems_distrib = Distribution('M', 1)

    # распределение времени создания защиты
    noise_protection_distrib = Distribution('M', 1)

    # распределение времени разведки
    intelligence_distrib = Distribution('Normal', [3, 1, 1, 30])

    # распределение времени постановки помех
    suppression_distrib = Distribution('Normal', [7, 1, 3, 30])

    # вероятность успешного подавления текущей помехой против текущего сигнала прд
    prob_suppression = 0.3

    # вероятность успешной защиты против текущей помехи
    prob_right_noise_protection = 0.6

    # Создаем имитационную модель РЭБ
    model = Model(prd_distrib, ems_distrib, noise_protection_distrib, intelligence_distrib, suppression_distrib,
              prob_suppression=prob_suppression, prob_right_noise_protection=prob_right_noise_protection,
              model_num=1000)

    # Запускаем имитационную модель
    model.run()

    print(f"Вероятность успешного подавления: {model.jam.success_num / model.prd.total_prd_num:0.3f}")
    print(f"Вероятность успешной передачи сигнала: {model.prd.success_num / model.prd.total_prd_num:0.3f}")

    plt.hist(create_normal_bins(prd_distrib.params, size=10000), bins=15, density=True, label="без РЭБ",
         alpha=0.7)
    plt.hist(model.prd.prd_times, bins=15, density=True, label="в условиях РЭБ", alpha=0.7)
    plt.legend()
    plt.title("Время передачи сигнала")
    plt.xlabel('t, сек')
    plt.show()
