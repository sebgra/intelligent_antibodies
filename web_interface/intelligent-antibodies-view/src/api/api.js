import axios from 'axios'
import generate from './generate'

function setAxiosInterceptors (axiosInstance) {
  axiosInstance.interceptors.request.use((config) => {
    return config
  })
  axiosInstance.interceptors.response.use((res) => {
    return res
  })
  return axiosInstance
}

const instance = setAxiosInterceptors(
  axios.create({
    baseURL: 'http://localhost:5000'
  })
)

export default {
  generate: generate(instance)
}