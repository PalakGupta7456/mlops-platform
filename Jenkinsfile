// Jenkins pipeline mirroring GitHub Actions CI + CD:
// lint -> test -> train gate -> build -> scan -> push -> deploy to Kubernetes (main only, manual approval)
pipeline {
  agent any
  environment {
    IMAGE = "ghcr.io/palakgupta7456/gridops"
    MLFLOW_DISABLE_AGENT_HINT = "1"
  }
  options { timestamps(); timeout(time: 45, unit: 'MINUTES'); disableConcurrentBuilds() }
  stages {
    stage('Setup') {
      steps { sh 'python3 -m venv .venv && . .venv/bin/activate && pip install -q -e ".[train,api,dev]"' }
    }
    stage('Lint') { steps { sh '. .venv/bin/activate && ruff check .' } }
    stage('Test') {
      steps { sh '. .venv/bin/activate && pytest --junitxml=reports/junit.xml' }
      post { always { junit 'reports/junit.xml' } }
    }
    stage('Train + quality gate') {
      steps {
        sh '''. .venv/bin/activate
              python -m gridops.pipeline --synthetic
              python -c "import json; m=json.load(open('state/serving/model.json'))['metrics']; assert m['skill_vs_persistence']>0, m"'''
      }
    }
    stage('Build image') { steps { sh 'docker build -t $IMAGE:$GIT_COMMIT .' } }
    stage('Scan image') { steps { sh 'docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy:0.56.2 image --severity HIGH,CRITICAL --exit-code 0 $IMAGE:$GIT_COMMIT' } }
    stage('Push') {
      when { branch 'main' }
      steps {
        withCredentials([usernamePassword(credentialsId: 'ghcr-creds', usernameVariable: 'U', passwordVariable: 'P')]) {
          sh 'echo $P | docker login ghcr.io -u $U --password-stdin && docker push $IMAGE:$GIT_COMMIT'
        }
      }
    }
    stage('Deploy to Kubernetes') {
      when { branch 'main' }
      steps {
        input message: "Roll out ${env.GIT_COMMIT.take(7)} to production?"
        withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
          sh '''kubectl apply -f deploy/k8s/
                kubectl -n gridops set image deployment/gridops-api api=$IMAGE:$GIT_COMMIT
                kubectl -n gridops rollout status deployment/gridops-api --timeout=180s || (kubectl -n gridops rollout undo deployment/gridops-api && exit 1)'''
        }
      }
    }
  }
  post { always { cleanWs() } }
}
