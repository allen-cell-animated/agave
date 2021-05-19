provider "aws" {
    region = "us-west-2"
    profile = "animatedcell"
}

resource "aws_ecr_repository" "agave_ecr_repo" {
    name = "agave-ecr-repo"
}

# 1. Retrieve an authentication token and authenticate your Docker client to your registry.
# Use the AWS CLI: (DMT: use --profile animatedcell in the aws command)
# aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 769425812337.dkr.ecr.us-west-2.amazonaws.com

# Note: If you receive an error using the AWS CLI, make sure that you have the latest version of the AWS CLI and Docker installed.
# 2. Build your Docker image using the following command. For information on building a Docker file from scratch see the instructions here . You can skip this step if your image is already built:
# docker build -t agave-ecr-repo .

# 3. After the build completes, tag your image so you can push the image to this repository:
# docker tag agave-ecr-repo:latest 769425812337.dkr.ecr.us-west-2.amazonaws.com/agave-ecr-repo:latest

# 4. Run the following command to push this image to your newly created AWS repository:
# docker push 769425812337.dkr.ecr.us-west-2.amazonaws.com/agave-ecr-repo:latest

resource "aws_ecs_cluster" "agave_cluster" {
  name = "agave-cluster" # Naming the cluster
}

resource "aws_ecs_task_definition" "agave_task" {
  family                   = "agave-task" # Naming our first task
  container_definitions    = <<DEFINITION
  [
    {
      "name": "agave-task",
      "image": "${aws_ecr_repository.agave_ecr_repo.repository_url}",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 1235,
          "hostPort": 1235
        }
      ],
      "memory": 512,
      "cpu": 4096,
      "gpu": 1
    }
  ]
  DEFINITION
  requires_compatibilities = ["EC2"]
  network_mode             = "host"
  execution_role_arn       = "${aws_iam_role.ecsTaskExecutionRole.arn}"
}

resource "aws_iam_role" "ecsTaskExecutionRole" {
  name               = "ecsTaskExecutionRole"
  assume_role_policy = "${data.aws_iam_policy_document.assume_role_policy.json}"
}

data "aws_iam_policy_document" "assume_role_policy" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "ecsTaskExecutionRole_policy" {
  role       = "${aws_iam_role.ecsTaskExecutionRole.name}"
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_ecs_service" "agave_service" {
  name            = "agave-service"                             # Naming our first service
  cluster         = "${aws_ecs_cluster.agave_cluster.id}"             # Referencing our created Cluster
  task_definition = "${aws_ecs_task_definition.agave_task.arn}" # Referencing the task our service will spin up
  launch_type     = "EC2"
  desired_count   = 1

  network_configuration {
    subnets          = ["${aws_default_subnet.default_subnet_a.id}", "${aws_default_subnet.default_subnet_b.id}", "${aws_default_subnet.default_subnet_c.id}"]
    assign_public_ip = true                                                # Providing our containers with public IPs
    security_groups  = ["${aws_security_group.service_security_group.id}"] # Setting the security group
  }
}

# Providing a reference to our default VPC
resource "aws_default_vpc" "default_vpc" {
}

# Providing a reference to our default subnets
resource "aws_default_subnet" "default_subnet_a" {
  availability_zone = "us-west-2"
}

resource "aws_default_subnet" "default_subnet_b" {
  availability_zone = "us-west-2"
}

resource "aws_default_subnet" "default_subnet_c" {
  availability_zone = "us-west-2"
}

resource "aws_security_group" "service_security_group" {
    # SSH access from anywhere
    ingress {
      from_port = 22
      to_port = 22
      protocol = "tcp"
      cidr_blocks = [
        "0.0.0.0/0"]
    }
    # SSH access from anywhere
    ingress {
      from_port = 1235
      to_port = 1235
      protocol = "tcp"
      cidr_blocks = [
        "0.0.0.0/0"]
    }
#   ingress {
#     from_port = 0
#     to_port   = 0
#     protocol  = "-1"
#     # Only allowing traffic in from the load balancer security group
#     # security_groups = ["${aws_security_group.load_balancer_security_group.id}"]
#   }

  egress {
    from_port   = 0 # Allowing any incoming port
    to_port     = 0 # Allowing any outgoing port
    protocol    = "-1" # Allowing any outgoing protocol 
    cidr_blocks = ["0.0.0.0/0"] # Allowing traffic out to all IP addresses
  }
}

# g3s.xlarge
# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html

# aws ssm get-parameters --names /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended
# https://us-west-2.console.aws.amazon.com/systems-manager/parameters/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id/description?region=us-west-2#
# Name
# /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id
# Value
# ami-0a8c654409fed3d9a