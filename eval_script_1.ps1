# Define dataset and segmentation type pairs
$pairs = @(
    @{ dataset_name = "optic_disc"; seg_type = "od" },
    @{ dataset_name = "optic_disc"; seg_type = "cup_only" },
    @{ dataset_name = "vessel"; seg_type = "od" }
)

# Define ranks
$ranks = @(32, 512, 1024)

# Define evaluation modes
$eval_modes = 0..6

# Loop through each pair of dataset and segmentation type
foreach ($pair in $pairs) {
    $dataset_name = $pair.dataset_name
    $seg_type = $pair.seg_type

    # Loop through each rank
    foreach ($rank in $ranks) {
        $alpha = 2 * $rank
        $output = "$seg_type"
        $checkpointpath = "checkpoints\fundus_${dataset_name}_${seg_type}_sam2_l_r${rank}_a${alpha}_best.ckpt"

        # Loop through evaluation modes
        foreach ($eval_mode in $eval_modes) {
            # Construct the command
            $command = "python3 test.py " +
                       "--lora_rank=$rank " +
                       "--lora_alpha=$alpha " +
                       "--checkpoint_path=$checkpointpath " +
                       "--output_name=$output " +
                       "--eval_mode=$eval_mode " +
                       "--seg_type=$seg_type " +
                       "--dataset_name=$dataset_name"

            # Echo the command
            Write-Output "Running command: $command"

            # Run the command
            Invoke-Expression $command
        }
    }
}
