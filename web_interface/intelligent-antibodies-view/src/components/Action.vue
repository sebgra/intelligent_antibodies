<template>
  <div class="action-wrapper">
    <button id='button' :disabled="!activated" class="action-button" @click="launchGen">
      Launch generation
    </button>
  </div>
</template>
<script>
export default {
  name:'Action',
  props: {
    activated: {
      type: Boolean
    }
  },
  methods: {
    launchGen() {
      console.log("Launching generation")
    }
  }
}
</script>
<style scoped>
.action-wrapper {
  display: flex;
  min-height: 1rem;
  width: 100%;
  justify-content: center;

}
.action-button {
  margin: auto;
  margin-top: 1rem;
  color: #38a3a5;
  border-color: #38a3a5;
  border: 1px solid;
}
.action-button:disabled {
  color: grey
}
</style>
