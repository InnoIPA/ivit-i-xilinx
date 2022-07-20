import time

class Timer():
    
    def __init__(self):
        
        self.t_start = time.time()          # 實例化的時間: 通常作為起始時間
        self.t_iter_start = self.t_start    # 迭代用的起始時間，用於計算多個階段所耗費的時間
        self.t_iter_end  = None             # 迭代用的結束時間，用於計算多個階段所耗費的時間

        self.t_index = 0                    # 紀錄第N個階段
        self.t_cost = None                  # 每個階段花費的時間

    def upd_iter_start(self):
        self.t_iter_start = time.time()

    def upd_iter_index(self):
        self.t_index+=1

    def get_start_time(self):
        return self.t_start

    def get_iter_time(self):
        return self.t_iter_start

    def get_dur_time(self):
        self.t_iter_end = time.time()
        self.t_cost = round(abs(self.t_iter_end-self.t_iter_start), 3)
        self.t_iter_start = time.time()
        return self.t_cost

    def get_cost_time(self):
        self.t_iter_end = time.time()
        return round(abs(self.t_iter_end-self.t_start), 3)

if __name__ == '__main__':
    
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    )

    logging.info('Initialize Timer')
    timer = Timer()

    time.sleep(3)
    logging.info('Initialize cost time: {}s'.format(timer.get_cost_time()))

    time.sleep(3)
    logging.info('Initialize cost time: {}s'.format(timer.get_cost_time()))