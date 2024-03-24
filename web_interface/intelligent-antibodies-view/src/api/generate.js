const baseURL = 'api/generate/'

export default (axiosInstance) => {
  const getGenerateData = async (sequence) => {
    try {
      const result = await axiosInstance.get(baseURL + sequence)
      return result.data
    } catch (error) {
      console.error(error)
      throw error
    }
  }

  return {
    getGenerateData
  }
}