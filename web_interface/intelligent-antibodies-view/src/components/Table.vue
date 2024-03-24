<template>
  <div class="table">
    <div class="table_header">
      <div class="table_sequence"><b class="weight">Sequence</b></div>
      <div class="table_button"><b class="weight">Action</b></div>
    </div>
    <div :key="line[0]" class="table_line" v-for="line in datas">
      <div class="table_sequence">
        {{ line[0] }}
      </div>
      <div class="table_button">
        <button :key="line[0]" @click="gotodisplay">View</button>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name:'Table',
  props: {
    datas: Array
  },
  data() {
      return {
      }
    },
  methods: {
    gotodisplay(event) {
      const sequence = event.target.__vnode.key
      this.$router.push({ path: '/display/' + sequence})
    }
  },
}
</script>


<style scoped>
.table {
  display: block;
  height:fit-content;
  border: 1px solid lightgrey;
  border-radius: 5px;
  justify-content: left;
  overflow:hidden;
}
.table_header {
  display: flex;
  justify-content: space-between;
  background-color: #409d9e;
  color: white;
  border-radius: 5px 5px 0 0;
}
.weight {
  font-weight: 700;
}
.table_line {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
  overflow: hidden;
}
.table_line:last-child{
  border-bottom: none
}
.table_line:hover {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
  background-color: lightgrey
}
.table_sequence {
  width: 90%;
  border-right: 1px solid lightgrey;
  padding: 5px;
  overflow: inherit;
}
.table_button {
  width: 2rem;
  padding: 5px;
  margin:auto;
}
</style>
