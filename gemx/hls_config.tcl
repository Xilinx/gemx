#create_clock -period 250MHz -name default
#set_clock_uncertainty 1.080000
config_core DSP48 -latency 2 
#config_schedule -effort medium -verbose
#config_core DSP48 -latency 5
#catch {config_rtl -no_idle}
#catch {config_rtl -keep_function_name}
#config_rtl -flush_pipe_on_stall=false

config_compile -ignore_long_run_time


