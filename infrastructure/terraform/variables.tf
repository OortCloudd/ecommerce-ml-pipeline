variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "ecommerce-ml-cluster"
}

variable "data_bucket_name" {
  description = "Name of the S3 bucket for data storage"
  type        = string
  default     = "ecommerce-ml-pipeline-data"
}
