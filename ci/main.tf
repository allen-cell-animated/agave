#########################################
## EXPERIMENTAL, Not ready for production
#########################################

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.3"
    }
  }
}
provider "aws" {
  region = "us-west-2"
  # this is specific to the way MY (danielt) credentials are stored for now.
  profile = "animatedcell"
}

# generic default
resource "aws_default_vpc" "default_vpc" {
}

# generic default
resource "aws_default_subnet" "default_subnet_a" {
  availability_zone = "us-west-2a"
}

# generic default
resource "aws_default_subnet" "default_subnet_b" {
  availability_zone = "us-west-2b"
}

# generic default
resource "aws_default_subnet" "default_subnet_c" {
  availability_zone = "us-west-2c"
}

# we want to allow access over
# port 22 for SSH,
# port 443 for the docker pull,
# and the application ports (for agave, 1235)
resource "aws_security_group" "agave_security_group" {
  # SSH access from anywhere
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # agave websocket access from anywhere
  ingress {
    from_port   = 1235
    to_port     = 1235
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # https for docker image
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  #   ingress {
  #     from_port = 0
  #     to_port   = 0
  #     protocol  = "-1"
  #     # Only allowing traffic in from the load balancer security group
  #     # security_groups = ["${aws_security_group.load_balancer_security_group.id}"]
  #   }

  egress {
    from_port   = 0             # Allowing any incoming port
    to_port     = 0             # Allowing any outgoing port
    protocol    = "-1"          # Allowing any outgoing protocol
    cidr_blocks = ["0.0.0.0/0"] # Allowing traffic out to all IP addresses
  }
}

# create an IAM role for the instances
data "aws_iam_policy_document" "agave_ecs_agent" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "agave_ecs_agent" {
  name               = "agave-ecs-agent"
  assume_role_policy = data.aws_iam_policy_document.agave_ecs_agent.json
}
resource "aws_iam_role_policy_attachment" "agave_ecs_agent" {
  role       = aws_iam_role.agave_ecs_agent.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}
resource "aws_iam_instance_profile" "agave_instance_profile" {
  name = "agave-instance-profile"
  role = aws_iam_role.agave_ecs_agent.name
}


# create an autoscaling group and tie it to an ECS cluster/service

# (1) the lightest/cheapest GPU server is this one:

# g3s.xlarge
# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html

# (2) this is the AMI that has gpu+ECS compatibility:

# aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended --region us-west-2
# https://us-west-2.console.aws.amazon.com/systems-manager/parameters/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id/description?region=us-west-2#
# Name
# /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id
# Value
# ami-01b66d92709ccc106

# {
#     "Parameters": [
#         {
#             "Name": "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended",
#             "Type": "String",
#             "Value": "{\"ecs_agent_version\":\"1.72.0\",\"ecs_runtime_version\":\"Docker version 20.10.23\",\"image_id\":\"ami-01b66d92709ccc106\",\"image_name\":\"amzn2-ami-ecs-gpu-hvm-2.0.20230606-x86_64-ebs\",\"image_version\":\"2.0.20230606\",\"os\":\"Amazon Linux 2\",\"schema_version\":1,\"source_image_name\":\"amzn2-ami-minimal-hvm-2.0.20230530.0-x86_64-ebs\"}",
#             "Version": 111,
#             "LastModifiedDate": 1686668971.933,
#             "ARN": "arn:aws:ssm:us-west-2::parameter/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended",
#             "DataType": "text"
#         }
#     ],
#     "InvalidParameters": []
# }

# I manually created the Key Pair "agave-ecs"
resource "aws_launch_configuration" "ecs_launch_config" {
  image_id             = "ami-0c55b159cbfafe1f0"
  iam_instance_profile = aws_iam_instance_profile.agave_instance_profile.name
  security_groups      = [aws_security_group.agave_security_group.id]
  user_data            = "#!/bin/bash\necho ECS_CLUSTER=agave-cluster >> /etc/ecs/ecs.config"
  instance_type        = "g3s.xlarge"
  key_name             = "agave-ecs"
  lifecycle {
    create_before_destroy = true
  }
}
resource "aws_autoscaling_group" "failure_analysis_ecs_asg" {
  name                      = "asg"
  vpc_zone_identifier       = [aws_default_subnet.default_subnet_a.id]
  launch_configuration      = aws_launch_configuration.ecs_launch_config.name
  desired_capacity          = 1
  min_size                  = 1
  max_size                  = 1
  health_check_grace_period = 300
  health_check_type         = "EC2"
  lifecycle {
    create_before_destroy = true
  }
  tag {
    key                 = "AmazonECSManaged"
    value               = true
    propagate_at_launch = true
  }
}
resource "aws_ecs_capacity_provider" "agave_capacity_provider" {
  name = aws_autoscaling_group.failure_analysis_ecs_asg.name

  auto_scaling_group_provider {
    auto_scaling_group_arn = aws_autoscaling_group.failure_analysis_ecs_asg.arn
  }
  lifecycle {
    create_before_destroy = true
  }
}
resource "aws_ecs_cluster" "agave_cluster" {
  name               = "agave-cluster" # Naming the cluster
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_ecs_cluster_capacity_providers" "agave_capacity_providers" {
  cluster_name = aws_ecs_cluster.agave_cluster.name
  capacity_providers = [aws_ecs_capacity_provider.agave_capacity_provider.name]
  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.agave_capacity_provider.name
    base              = 1
    weight            = 100
  }
}

resource "aws_ecr_repository" "agave_ecr_repo" {
  name = "agave-ecr-repo"
}

resource "aws_ecs_task_definition" "agave_task" {
  family                = "agave-task" # Naming our first task
  container_definitions = <<DEFINITION
  [
    {
      "name": "agave-task",
      "image": "${aws_ecr_repository.agave_ecr_repo.repository_url}:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 1235,
          "hostPort": 1235
        }
      ],
      "memory": 4096,
      "cpu": 4096,
      "gpu": 1,
      "resourceRequirements": [
        {
          "type" : "GPU",
          "value" : "1"
        }
      ]
    }
  ]
  DEFINITION
}


resource "aws_ecs_service" "agave_service" {
  name                       = "agave-service"                        # Naming our first service
  cluster                    = aws_ecs_cluster.agave_cluster.id       # Referencing our created Cluster
  task_definition            = aws_ecs_task_definition.agave_task.arn # Referencing the task our service will spin up
  launch_type                = "EC2"
  desired_count              = 1
  deployment_maximum_percent = 200
}

# to force an image update:

# aws ecs update-service --cluster agave-cluster --service agave-service --force-new-deployment --region us-west-2 --profile animatedcell

# 1. Retrieve an authentication token and authenticate your Docker client to your registry.
# Use the AWS CLI: (DMT: use --profile animatedcell in the aws command, specific to how my credentials are stored)
#
# aws ecr get-login-password --region us-west-2 --profile animatedcell | docker login --username AWS --password-stdin 769425812337.dkr.ecr.us-west-2.amazonaws.com


# Note: If you receive an error using the AWS CLI, make sure that you have the latest version of the AWS CLI and Docker installed.
# 2. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:
#
# docker build -t agave-ecr-repo .

# 3. After the build completes, tag your image so you can push the image to this repository:
#
# docker tag agave-ecr-repo:latest 769425812337.dkr.ecr.us-west-2.amazonaws.com/agave-ecr-repo:latest

# 4. Run the following command to push this image to your newly created AWS repository:
#
# docker push 769425812337.dkr.ecr.us-west-2.amazonaws.com/agave-ecr-repo:latest


