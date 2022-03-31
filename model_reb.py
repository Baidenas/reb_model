import rand_destribution as rd
import numpy as np
import math
from tqdm import tqdm

# параметры  тракта приемо-передачи
PRD_PRM_PARAMS = {
    'source':
    {
        'params': 1.0,  # параметры распределения, для экспоненциально - лямбда
        'types': 'M'    # тип распределения. М - экспоненциальное
    },
    'buffer': 100,
    'prd':
    {
        'params': [1, 1],  # среднее, СКО
        'types': 'Normal'
    },
    'em_env_control':
    {
        'params': 1.0/45,
        'types': 'M'
    },
    'anti_jam_measures':
    {
        'params': [15, 5],   # среднее и полуинтервал влево и вправо
        'types': 'Uniform',  # равномерное
        'probs': [0.9, 0.1]  # вероятности после завершения [успех, неудача]
    }
}

REB_PARAMS = {
    'intelligence':
    {
        'params': 1.0/12,
        'types': 'M',
        'probs': [0.9, 0.1]
    },
    'making_noise':  # объединил действия после успешной разведки в одно - постановка помех. Если нужно, можно добавить
    {
        'params': [10, 3],
        'types': 'Normal',
        'probs': [0.9, 0.1]
    }
}


class SetModelException(Exception):

    def __str__(self, text):
        return text


class Task:
    """
    Пакет на передачу
    """
    id = 0

    def __init__(self, arr_time):
        """
        :param arr_time: Момент начала передачи
        """
        self.arr_time = arr_time

        self.start_waiting_time = -1

        self.wait_time = 0

        Task.id += 1
        self.id = Task.id

    def __str__(self):
        res = "Task #" + str(self.id) + "\n"
        res += "\tArrival moment: " + "{0:8.3f}".format(self.arr_time)
        return res


class REB_model:
    """
        Модель РЭБ
    """
    def __init__(self,
                 prm_prd_params=PRD_PRM_PARAMS,  # передача на вход параметров, по умолчанию заданы вверху
                 reb_params=REB_PARAMS
                 ):
        self.prm_prd_params = prm_prd_params
        self.reb_params = reb_params
        self.is_channel_free = True
        self.is_suppressed = False
        self.source = {}
        self.prd = {}
        self.em_env_control = {}
        self.anti_jam_measures = {}
        self.intelligence = {}
        self.making_noise = {}
        self.queue = []
        self.task_on_prd = None

        self.t_tek = 0
        self.task_in_system_count = 0
        self.taked = 0
        self.arrived = 0
        self.task_rejected_count = 0
        self.end_prd_task_count = 0
        self.suppressions_count = 0

        self.prm_prd_moments = [0, 0, 0]
        self.wait_moments = [0, 0, 0]

        self.times = {}
        self.times['arrival_time'] = 0
        self.times['end_prd_time'] = 1e10
        self.times['end_em_env_control_time'] = 1e10
        self.times['end_intelligence_time'] = 1e10
        self.times['end_making_noise_time'] = 1e10
        self.times['end_suppressing_time'] = 1e10
        self.times['end_anti_jam_measures_time'] = 1e10

        # PRM PRD PARAMETRES:
        self.set_params(self.source, prm_prd_params['source']['params'], prm_prd_params['source']['types'])
        self.times['arrival_time'] = self.source['dist'].generate()

        if prm_prd_params['buffer']:
            self.is_buffer_set = True
            self.buffer_length = prm_prd_params['buffer']

        self.set_params(self.prd, prm_prd_params['prd']['params'], prm_prd_params['prd']['types'])
        self.set_params(self.em_env_control, prm_prd_params['em_env_control']['params'], prm_prd_params['em_env_control']['types'])

        self.set_params(self.anti_jam_measures, prm_prd_params['anti_jam_measures']['params'],
                        prm_prd_params['anti_jam_measures']['types'])
        self.anti_jam_measures['probs'] = prm_prd_params['anti_jam_measures']['probs']

        # REB PARAMETRES:
        self.set_params(self.intelligence, reb_params['intelligence']['params'],
                        reb_params['intelligence']['types'])
        self.intelligence['probs'] = reb_params['intelligence']['probs']
        self.start_intelligence()

        self.set_params(self.making_noise, reb_params['making_noise']['params'],
                        reb_params['making_noise']['types'])
        self.making_noise['probs'] = reb_params['making_noise']['probs']

    def __str__(self):
        res = "Модель РЭБ \n"
        if self.is_buffer_set != None:
            res += "Размер буффера " + str(self.buffer_length)+"\n"
        res += "Текущее время " + "{0:8.3f}".format(self.t_tek) + "\n"

        res += "Начальные моменты времени передачи:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.prm_prd_moments[i])
        res += "\n"

        res += "Начальные моменты времени ожидания:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.wait_moments[i])
        res += "\n"

        res += "Всего пакетов: " + str(self.arrived) + "\n"
        if self.is_buffer_set != None:
            res += "Потеряно: " + str(self.task_rejected_count) + "\n"
        res += "Начали передачу: " + str(self.taked) + "\n"
        res += "Закончили передачу: " + str(self.end_prd_task_count) + "\n"
        res += "Находятся в процессе:" + str(self.task_in_system_count) + "\n"
        res += "В буффере " + str(len(self.queue)) + "\n"
        res += "Кол-во подавлений " + str(self.suppressions_count) + "\n"

        return res

    def set_params(self, self_params, params_new, types_new):
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
        self_params['params'] = params_new
        self_params['types'] = types_new

        self.is_set_source_params = True

        if self_params['types'] == "M":
            self_params['dist'] = rd.Exp_dist(self_params['params'])
        elif self_params['types'] == "H":
            self_params['dist'] = rd.H2_dist(self_params['params'])
        elif self_params['types'] == "E":
            self_params['dist'] = rd.Erlang_dist(self_params['params'])
        elif self_params['types'] == "C":
            self_params['dist'] = rd.Cox_dist(self_params['params'])
        elif self_params['types'] == "Pa":
            self_params['dist'] = rd.Pareto_dist(self_params['params'])
        elif self_params['types'] == "Normal":
            self_params['dist'] = rd.Normal_dist(self_params['params'])
        elif self_params['types'] == "Gamma":
            self_params['dist'] = rd.Gamma(self_params['params'])
        elif self_params['types'] == "Uniform":
            self_params['dist'] = rd.Uniform_dist(self_params['params'])
        elif self_params['types'] == "D":
            self_params['dist'] = rd.Det_dist(self_params['params'])
        else:
            raise SetModelException("Неправильно задан тип распределения источника. Варианты М, Н, Е, С, Pa, Uniform")

    def refresh_prm_prd_time_stat(self, new_a):
        for i in range(3):
            self.prm_prd_moments[i] = self.prm_prd_moments[i] * (1.0 - (1.0 / self.end_prd_task_count)) + math.pow(new_a, i + 1) / self.end_prd_task_count

    def refresh_wait_time_stat(self, new_a):
        for i in range(3):
            self.wait_moments[i] = self.wait_moments[i] * (1.0 - (1.0 / self.taked)) + math.pow(new_a, i + 1) / self.taked


    def run(self, job_count=1000000):

        for i in tqdm(range(job_count)):
            min_time = 1e10
            event_name = 0
            for time in self.times:
                if self.times[time] < min_time:
                    min_time = self.times[time]
                    event_name = time

            if event_name == "arrival_time":
                self.start_prd(min_time)
            elif event_name == "end_prd_time":
                self.end_prd()
            elif event_name == "end_em_env_control_time":
                self.end_em_env_control()
            elif event_name == "end_intelligence_time":
                self.end_intelligence()
            elif event_name == "end_making_noise_time":
                self.end_making_noise()
            elif event_name == "end_anti_jam_measures_time":
                self.end_anti_jam_measures()


    def start_prd(self, arr_time):
        tsk = Task(arr_time)
        self.t_tek = arr_time
        self.times['arrival_time'] = self.source['dist'].generate() + self.t_tek

        self.arrived += 1

        if self.is_channel_free:
            self.times['end_prd_time'] = self.prd['dist'].generate() + self.t_tek
            self.task_on_prd = tsk
            self.task_in_system_count += 1

            self.taked += 1
            self.refresh_wait_time_stat(0)

            self.times['end_prd_time'] = self.source['dist'].generate() + self.t_tek

            self.is_channel_free = False
        else:

            if self.is_buffer_set:
                # max in system = 1 + buffer_length
                if len(self.queue) >= self.buffer_length:
                    self.task_rejected_count += 1
                else:
                    tsk.start_waiting_time = self.t_tek
                    self.queue.append(tsk)
                    self.task_in_system_count += 1
            else:
                tsk.start_waiting_time = self.t_tek
                self.task_in_system_count += 1
                self.queue.append(tsk)

    def end_prd(self):

        tsk = self.task_on_prd
        self.t_tek = self.times['end_prd_time']
        self.times['end_prd_time'] = 1e10
        self.end_prd_task_count += 1
        self.task_in_system_count -= 1
        self.is_channel_free = True
        self.refresh_prm_prd_time_stat(self.t_tek - tsk.arr_time)


        if len(self.queue) != 0:
            tsk = self.queue.pop(0)
            self.taked += 1
            tsk.wait_time += self.t_tek - tsk.start_waiting_time
            self.refresh_wait_time_stat(tsk.wait_time)
            self.times['end_prd_time'] = self.source['dist'].generate() + self.t_tek
            self.task_on_prd = tsk
            self.is_channel_free = False

    def on_suppression(self):
        self.suppressions_count += 1
        self.is_suppressed = True
        self.start_em_env_control()
        self.start_intelligence()

    def start_em_env_control(self):
        self.times['end_em_env_control_time'] = self.t_tek + self.em_env_control['dist'].generate()

    def end_em_env_control(self):
        self.t_tek = self.times['end_em_env_control_time']
        self.times['end_em_env_control_time'] = 1e10
        self.start_anti_jam_measures()

    def start_anti_jam_measures(self):
        self.times['end_anti_jam_measures_time'] = self.t_tek + self.anti_jam_measures['dist'].generate()

    def end_anti_jam_measures(self):
        self.t_tek = self.times['end_anti_jam_measures_time']
        self.times['end_anti_jam_measures_time'] = 1e10
        r = np.random.rand()
        if r < self.anti_jam_measures['probs'][0]:
            self.start_prd(self.t_tek)
        else:
            self.start_em_env_control()


    def start_intelligence(self):
        self.times['end_intelligence_time'] = self.t_tek + self.intelligence['dist'].generate()

    def end_intelligence(self):
        self.t_tek = self.times['end_intelligence_time']
        self.times['end_intelligence_time'] = 1e10
        if self.is_channel_free:
            self.start_intelligence()
        else:
            r = np.random.rand()
            if r < self.intelligence['probs'][0]:
                # разведка успешна
                self.start_making_noise()

            else:
                self.start_intelligence()

    def start_making_noise(self):
        self.times['end_making_noise_time'] = self.t_tek + self.making_noise['dist'].generate()

    def end_making_noise(self):
        self.t_tek = self.times['end_making_noise_time']
        self.times['end_making_noise_time'] = 1e10
        r = np.random.rand()
        if r < self.making_noise['probs'][0]:
            self.on_suppression()
        else:
            self.start_intelligence()


if __name__ == '__main__':

    # пример использования - создаем экземпляр класса и передаем параметры, если нужно подправить - они вверху
    model = REB_model(PRD_PRM_PARAMS, REB_PARAMS)

    model.run(1000000)  #  задаем кол-во пакетов на вход и запускаем

    print(model)  # вывод результатов