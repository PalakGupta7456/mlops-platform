# Serverless container hosting for the model API on AWS App Runner (scales to zero traffic cost),
# image in ECR, logs in CloudWatch. Applied by .github/workflows/cd.yml when AWS secrets exist.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    key    = "gridops/terraform.tfstate"
    region = "ap-south-1"
  }
}

provider "aws" { region = var.region }

variable "region" {
  type    = string
  default = "ap-south-1"
}
variable "image_tag" { type = string }

resource "aws_ecr_repository" "api" {
  name                 = "gridops"
  force_delete         = true
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy     = jsonencode({ rules = [{ rulePriority = 1, description = "keep 10", action = { type = "expire" }, selection = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 } }] })
}

resource "aws_iam_role" "apprunner_ecr" {
  name               = "gridops-apprunner-ecr"
  assume_role_policy = jsonencode({ Version = "2012-10-17", Statement = [{ Effect = "Allow", Principal = { Service = "build.apprunner.amazonaws.com" }, Action = "sts:AssumeRole" }] })
}

resource "aws_iam_role_policy_attachment" "apprunner_ecr" {
  role       = aws_iam_role.apprunner_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_apprunner_auto_scaling_configuration_version" "api" {
  auto_scaling_configuration_name = "gridops"
  min_size                        = 1
  max_size                        = 3
  max_concurrency                 = 50
}

resource "aws_apprunner_service" "api" {
  service_name                   = "gridops-api"
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.api.arn
  source_configuration {
    auto_deployments_enabled = false
    authentication_configuration { access_role_arn = aws_iam_role.apprunner_ecr.arn }
    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      image_configuration { port = "8000" }
    }
  }
  instance_configuration {
    cpu    = "0.25 vCPU"
    memory = "0.5 GB"
  }
  health_check_configuration {
    protocol = "HTTP"
    path     = "/health"
  }
}

output "api_url" { value = "https://${aws_apprunner_service.api.service_url}" }

output "ecr_url" { value = aws_ecr_repository.api.repository_url }
