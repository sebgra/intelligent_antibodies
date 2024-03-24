const baseURL = 'api/structure/'

export default (axiosInstance) => {
  const getId = async (sequence) => {
    try {
      const result = await axiosInstance.get(baseURL + 'align/' + sequence)
      return result.data
    } catch (error) {
      console.error(error)
      throw error
    }
  }

  const getPDB = async (id) => {
    try {
      const result = await axiosInstance.get(baseURL + 'retrieve/' + id)
      return result.data
    } catch (error) {
      console.error(error)
      throw error
    }
  }

  return {
    getId,
    getPDB
  }
}