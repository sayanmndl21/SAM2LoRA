# Uncomment to activate the virtual environment
# & "C:\Users\Sayan Mandal\workspace\venv\sam2\Scripts\activate"

# Define variables
$checkpointpath = "checkpoints\fundus_optic_disc_cup_only_sam2_l_r256_a512_best.ckpt"
$rank = 256
$alpha = 512
$seg_type = "cup_only"
$output = "$seg_type"
$dataset_name = "optic_disc"

# Run the Python script in different evaluation modes
for ($eval_mode = 0; $eval_mode -le 6; $eval_mode++) {
    # Construct the command
    $command = "python3 test.py " +
            "--lora_rank=$rank " +
            "--lora_alpha=$alpha " +
            "--checkpoint_path=$checkpointpath " +
            "--output_name=$output " +
            "--eval_mode=$eval_mode " +
            "--seg_type=$seg_type " +
            "--dataset_name=$dataset_name "
    # Echo the command
    Write-Output "Running command: $command"
    
    # Run the command
    Invoke-Expression $command    
}

