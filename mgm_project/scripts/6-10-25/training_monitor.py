#!/usr/bin/env python3


"""
Real-time Training Monitor
Continuously monitors training progress, memory usage, and system health
"""

import os 
import time 
import subprocess 
import psutil 
import logging 
from datetime import datetime 
from pathlib import Path 

logging .basicConfig (level =logging .INFO ,format ='%(asctime)s - %(message)s')
logger =logging .getLogger (__name__ )

class TrainingMonitor :
    def __init__ (self ,check_interval =30 ):
        self .check_interval =check_interval 
        self .last_log_size =0 
        self .start_time =time .time ()
        self .error_count =0 
        self .max_errors =5 

    def check_gpu_status (self ):
        """Check GPU memory and utilization"""
        try :
            result =subprocess .run (['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'],
            capture_output =True ,text =True ,timeout =10 )

            if result .returncode ==0 :
                lines =result .stdout .strip ().split ('\n')
                for i ,line in enumerate (lines ):
                    memory_used ,memory_total ,gpu_util ,temp =line .split (', ')
                    memory_percent =(int (memory_used )/int (memory_total ))*100 

                    logger .info (f"🖥️  GPU {i }: {memory_percent :.1f}% memory ({memory_used }MB/{memory_total }MB), "
                    f"{gpu_util }% util, {temp }°C")


                    if memory_percent >90 :
                        logger .warning (f"🚨 HIGH GPU MEMORY: {memory_percent :.1f}%")
                    elif memory_percent >80 :
                        logger .warning (f"⚠️  GPU memory getting high: {memory_percent :.1f}%")


                    if int (temp )>80 :
                        logger .warning (f"🌡️  High GPU temperature: {temp }°C")

        except Exception as e :
            logger .error (f"❌ Failed to get GPU status: {e }")

    def check_training_process (self ):
        """Check if training process is running"""
        try :
            result =subprocess .run (['pgrep','-f','run_flagship_production.py'],
            capture_output =True ,text =True )

            if result .returncode ==0 :
                pids =result .stdout .strip ().split ('\n')
                logger .info (f"✅ Training process running (PIDs: {', '.join (pids )})")


                for pid in pids :
                    if pid :
                        try :
                            proc =psutil .Process (int (pid ))
                            cpu_percent =proc .cpu_percent (interval =1 )
                            memory_info =proc .memory_info ()
                            memory_mb =memory_info .rss /1024 /1024 

                            logger .info (f"   PID {pid }: {cpu_percent :.1f}% CPU, {memory_mb :.1f}MB RAM")
                        except Exception as e :
                            logger .warning (f"   Could not get details for PID {pid }: {e }")

                return True 
            else :
                logger .error ("❌ Training process not found!")
                return False 

        except Exception as e :
            logger .error (f"❌ Failed to check training process: {e }")
            return False 

    def check_log_progress (self ):
        """Monitor log file for progress and errors"""
        log_files =['training_PROTECTED.log','training_WORKING.log','training.log']

        for log_file in log_files :
            if os .path .exists (log_file ):
                try :
                    stat =os .stat (log_file )
                    current_size =stat .st_size 

                    if current_size >self .last_log_size :

                        with open (log_file ,'r')as f :
                            f .seek (self .last_log_size )
                            new_content =f .read ()


                        lines =new_content .strip ().split ('\n')[-10 :]

                        for line in lines :
                            if 'CUDA out of memory'in line :
                                logger .error ("🚨 OOM ERROR detected in logs!")
                                self .error_count +=1 
                            elif 'NaN'in line or 'inf'in line :
                                logger .error ("🚨 NUMERICAL INSTABILITY detected!")
                                self .error_count +=1 
                            elif 'Step'in line and 'Loss'in line :
                                logger .info (f"📈 Training progress: {line .strip ()}")
                            elif 'Epoch'in line :
                                logger .info (f"🔄 {line .strip ()}")
                            elif 'ERROR'in line .upper ():
                                logger .error (f"❌ {line .strip ()}")
                                self .error_count +=1 

                        self .last_log_size =current_size 
                        logger .info (f"📄 Log file updated: {log_file } ({current_size } bytes)")
                        return True 

                except Exception as e :
                    logger .error (f"❌ Error reading log file {log_file }: {e }")

        logger .warning ("⚠️  No active log files found")
        return False 

    def check_disk_space (self ):
        """Check available disk space"""
        try :
            usage =psutil .disk_usage ('.')
            free_gb =usage .free /(1024 **3 )
            total_gb =usage .total /(1024 **3 )
            used_percent =(usage .used /usage .total )*100 

            logger .info (f"💾 Disk: {used_percent :.1f}% used, {free_gb :.1f}GB free of {total_gb :.1f}GB")

            if free_gb <5 :
                logger .error ("🚨 LOW DISK SPACE: Less than 5GB free!")
            elif free_gb <10 :
                logger .warning ("⚠️  Disk space getting low: {free_gb:.1f}GB free")

        except Exception as e :
            logger .error (f"❌ Failed to check disk space: {e }")

    def check_system_health (self ):
        """Overall system health check"""
        try :

            cpu_percent =psutil .cpu_percent (interval =1 )


            memory =psutil .virtual_memory ()
            memory_percent =memory .percent 
            memory_available_gb =memory .available /(1024 **3 )


            try :
                load_avg =os .getloadavg ()
                load_str =f", load: {load_avg [0 ]:.2f}"
            except :
                load_str =""

            logger .info (f"🖥️  System: {cpu_percent :.1f}% CPU, {memory_percent :.1f}% RAM "
            f"({memory_available_gb :.1f}GB free){load_str }")


            if memory_percent >90 :
                logger .warning ("🚨 HIGH SYSTEM MEMORY USAGE!")
            elif memory_percent >80 :
                logger .warning ("⚠️  System memory getting high")

            if cpu_percent >95 :
                logger .warning ("🚨 HIGH CPU USAGE!")

        except Exception as e :
            logger .error (f"❌ Failed to check system health: {e }")

    def run_health_check (self ):
        """Run comprehensive health check"""
        runtime =time .time ()-self .start_time 
        runtime_str =f"{int (runtime //3600 ):02d}:{int ((runtime %3600 )//60 ):02d}:{int (runtime %60 ):02d}"

        logger .info (f"\n{'='*60 }")
        logger .info (f"🏥 HEALTH CHECK - Runtime: {runtime_str }")
        logger .info (f"{'='*60 }")


        process_running =self .check_training_process ()


        self .check_system_health ()
        self .check_gpu_status ()
        self .check_disk_space ()


        log_progress =self .check_log_progress ()


        if self .error_count >=self .max_errors :
            logger .error (f"🚨 TOO MANY ERRORS ({self .error_count }). Consider stopping training!")

        if not process_running :
            logger .error ("🚨 TRAINING PROCESS STOPPED!")
            return False 

        logger .info (f"{'='*60 }\n")
        return True 

    def monitor (self ):
        """Main monitoring loop"""
        logger .info ("🚀 Starting training monitor...")
        logger .info (f"📊 Checking every {self .check_interval } seconds")

        try :
            while True :
                if not self .run_health_check ():
                    logger .error ("❌ Critical issues detected. Monitoring stopped.")
                    break 

                time .sleep (self .check_interval )

        except KeyboardInterrupt :
            logger .info ("⛔ Monitoring stopped by user")
        except Exception as e :
            logger .error (f"❌ Monitor crashed: {e }")

if __name__ =="__main__":
    monitor =TrainingMonitor (check_interval =30 )
    monitor .monitor ()
