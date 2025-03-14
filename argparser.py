import os
import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Argument parser for training hyperparameters")

    # Base path argument
    parser.add_argument('--basepath', type=str, default=os.getcwd(),
                        help="Base path for relative directories (default: current working directory)")

    # Model and training related arguments
    parser.add_argument('--checkpoint_path', type=str, required=False, default="./checkpoints",
                        help="Path to the model checkpoint")
    parser.add_argument('--output_path', type=str, default='./checkpoints',
                        help="Path to save the output")
    parser.add_argument('--output_name', type=str, required=False, default=None,
                        help="Name of the output")
    parser.add_argument('--eval_mode', type=int, choices=[0,1,2,3,4,5,6], default=0,
                        help="Eval mode (out of 6 categories)")
    parser.add_argument('--eval_region', type=str, choices=['general', 'center', 'periphery', 'density'], default='general',
                        help="Evaluates prompt generation based on region of interest")
    
    # Dataset related arguments
    parser.add_argument('--dataset_name', type=str, required=False, default='vessel',
                        help="Name of the dataset")
    parser.add_argument('--seg_type', type=str, choices=['od', 'cup', 'rim', 'cup_only'], default='od',
                        help="Segmentation Type")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for training (default: 8)")
    parser.add_argument('--use_transform', type=str, default="true",
                        help="Use transform on dataset")

    # LoRA related arguments
    parser.add_argument('--model_size', type=str, choices=['b','s','l'], required=False, default='l',
                        help="LoRA rank")
    parser.add_argument('--lora_rank', type=int, required=False, default=64,
                        help="LoRA rank")
    parser.add_argument('--lora_alpha', type=float, required=False, default=128,
                        help="LoRA alpha value")

    # Training options
    parser.add_argument('--high_res', action='store_true', default=True,
                        help="Whether to use high resolution (default: True)")
    parser.add_argument('--num_pos_points', default=200,
                        help="Number of positive points (default: 200)")
    parser.add_argument('--num_neg_points', default=0,
                        help="Number of negative points (default: 0)")
    parser.add_argument('--num_boxes', default=0,
                        help="Number of box prompts (default: 0)")
    parser.add_argument('--optimizer_type', type=str, choices=['sgd', 'adamw'], default='adamw',
                        help="Optimizer type (sgd or adamw)")
    parser.add_argument('--scheduler_type', type=str, default='cosinewarm',
                        help="Scheduler type (default: cosinewarm)")
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help="Gradient accumulation steps (default: 2)")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Learning rate (default: 0.0001)")
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help="Weight decay for optimizer (default: 3e-4)")
    parser.add_argument('--nesterov', action='store_true',
                        help="Use Nesterov momentum (default: False)")
    parser.add_argument('--step_size', type=int, default=1000,
                        help="Scheduler step size (default: 1000)")
    parser.add_argument('--momentum', type=float, default=1e-4,
                        help="Momentum for optimizer (default: 1e-4)")
    parser.add_argument('--number_steps', type=int, default=1000,
                        help="Total number of training steps (default: 1000)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of epochs (default: None)")

    # Logging and saving options
    parser.add_argument('--tensorboard_path', type=str, default='./runs',
                        help="Path to save tensorboard logs (default: ./runs)")
    parser.add_argument('--log_path', type=str, default=None,
                        help="Path to save logs (default: None)")
    parser.add_argument('--save_frequency', type=int, default=1000,
                        help="Frequency to save checkpoints (default: every 1000 steps)")

    return parser