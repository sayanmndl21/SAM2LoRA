# Uncomment to activate the virtual environment
# & "C:\Users\Sayan Mandal\workspace\venv\sam2\Scripts\activate"

# Define variables
$checkpointpath = "checkpoints\"
$dataset_name = "optic_disc"
$rank = 256
$alpha = 512
$seg_type="cup_only"
$output = "$seg_type"
$learning_rate=1e-4
$number_steps =120000
$num_pos_points = 5
$num_neg_points = 0
$eval_region = "general"
$batch_size =2
$accumulation_steps=4
$use_transform="True"
$save_frequency=10000

# Construct the command
$command = "python3 train.py " `
    + "--lora_rank=$rank " `
    + "--lora_alpha=$alpha " `
    + "--checkpoint_path=$checkpointpath " `
    + "--dataset_name=$dataset_name " `
    + "--learning_rate=$learning_rate " `
    + "--number_steps=$number_steps " `
    + "--output_name=$output " `
    + "--seg_type=$seg_type " `
    + "--num_pos_points=$num_pos_points " `
    + "--num_neg_points=$num_neg_points " `
    + "--eval_region=$eval_region " `
    + "--batch_size=$batch_size " `
    + "--accumulation_steps=$accumulation_steps " `
    + "--use_transform=$use_transform " `
    + "--save_frequency=$save_frequency"


# Echo the command
Write-Output "Running command: $command"

# Run the command
Invoke-Expression $command
