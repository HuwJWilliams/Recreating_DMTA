#Define arrays for the variables
n_cmpds_list=(50 50 50 50)
sel_methods=("rmp" "rmp" "rmp" "rmp")
start_iters=(1 1 1 1)
total_iters=(30 30 30 30)
run_dates=(20250307 20250307 20250307 20250307)
random_fracs=("0.05" "0.025" "0.25" "0.5")
extra_descriptions=("005" "0025" "025" "05")

# Generate and submit job scripts for each combination of variables
# for n_cmpds in "${n_cmpds_list[@]}"; do
#     for sel_method in "${sel_methods[@]}"; do
#         for start_iter in "${start_iters[@]}"; do
#             for total_iter in "${total_iters[@]}"; do
#                 for run_date in "${run_dates[@]}"; do
#                     for random_frac in "${random_fracs[@]}"; do

#                         # Create a unique job script
#                         job_script="run_${run_date}_${n_cmpds}_${sel_method}.sh"
                        
#                         # Replace placeholders in the template
#                         sed -e "s|<N_CMDS>|${n_cmpds}|g" \
#                             -e "s|<SEL_METHOD>|${sel_method}|g" \
#                             -e "s|<START_ITER>|${start_iter}|g" \
#                             -e "s|<TOTAL_ITERS>|${total_iter}|g" \
#                             -e "s|<RUN_DATE>|${run_date}|g" \
#                             -e "s|<RANDOM_FRAC>|${random_frac}|g" \
#                             run_template.sh > "$job_script"
                        
#                         # Submit the job script
#                         echo "Submitting $job_script..."
#                         sbatch "$job_script"
#                         mv $job_script ./job_scripts/
#                     done
#                 done
#             done
#         done
#     done
# done

# Ensure all arrays have the same length
array_length=${#n_cmpds_list[@]}

# Loop through the arrays
for ((i=0; i<array_length; i++)); do
    n_cmpds=${n_cmpds_list[$i]}
    sel_method=${sel_methods[$i]}
    start_iter=${start_iters[$i]}
    total_iter=${total_iters[$i]}
    run_date=${run_dates[$i]}
    random_frac=${random_fracs[$i]}
    extra_description=${extra_descriptions[$i]}

    # Create a unique job script
    job_script="run_${run_date}_${n_cmpds}_${sel_method}.sh"
    
    # Replace placeholders in the template
    sed -e "s|<N_CMDS>|${n_cmpds}|g" \
        -e "s|<SEL_METHOD>|${sel_method}|g" \
        -e "s|<START_ITER>|${start_iter}|g" \
        -e "s|<TOTAL_ITERS>|${total_iter}|g" \
        -e "s|<RUN_DATE>|${run_date}|g" \
        -e "s|<RANDOM_FRAC>|${random_frac}|g" \
        -e "s|<EXTRA_DESCRIPTION>|${extra_description}|g" \
        run_template.sh > "$job_script"
    
    # Submit the job script
    echo "Submitting $job_script..."
    sbatch "$job_script"
    mv $job_script ./job_scripts/
done