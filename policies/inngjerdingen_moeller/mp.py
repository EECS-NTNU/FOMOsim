from multiprocessing import Pool
import time

def sum_square(number):
    s = 0
    for i in range(number):
        s += i*i
    return s

def sum_square_with_mp(numbers):
    start_time = time.time()
    p=Pool() #default is all available cores
    result = p.map(sum_square, numbers)
    p.close()
    p.join()
    end_time = time.time()-start_time
    
    print(f"Processing {len(numbers)} numbers took {end_time} time using multiprocessing.")

def sum_square_no_mp(numbers):
    start_time = time.time()
    result = []
    for i in numbers:
        result.append(sum_square(i))
    end_time = time.time()-start_time
    print(f"Processing {len(numbers)} numbers took {end_time} time with no multiprocessing.")

if __name__ == '__main__':
    sum_square_with_mp(range(100))
    sum_square_no_mp(range(100))
    
    sum_square_with_mp(range(20000))
    sum_square_no_mp(range(20000))